# Copyright (C) 2026 M.R. Siavash Katebzadeh <mr@katebzadeh.xyz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import ctypes
import enum
import sys
from pathlib import Path

DSMI_SO_PATH = "/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so"
ASCEND_TOOLKIT_ROOT = Path("/usr/local/Ascend/ascend-toolkit/latest")


def _ascend_ml_library_path(root, name):
    candidates = (root / "lib64" / name, *root.glob("*/lib64/" + name))
    return next((path for path in candidates if path.is_file()), None)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super().__call__(*args, **kwargs)
        return Singleton._instances[cls]

    def clear(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass


class RetCode(enum.Enum):
    SUCCESS = 0
    FAILED = 1
    ARG_PATH_INVALID = 2
    ARG_EMPTY_STRING = 3
    ARG_SAPCE_STRING = 4
    ARG_ILLEGAL_STRING = 5
    ARG_NO_EXIST_DIR = 6
    ARG_NO_EXECUTABLE = 7
    ARG_NO_WRITABLE_PERMISSION = 8
    ARG_CREATE_DIR_FAILED = 9
    READ_FILE_FAILED = 10
    PERMISSION_FAILED = 11


class LoadSoType(metaclass=Singleton):
    def __init__(self):
        self.drvdsmi = None
        self.drvhal = None
        self.asendml = None
        self.ascend_trace = None
        self.env_type = ""

    @staticmethod
    def load_dll(so_name):
        try:
            dll = ctypes.cdll.LoadLibrary(so_name)
        except OSError as err:
            sys.stderr.write(f"OSError: {err}")
            return RetCode.FAILED
        return dll

    def get_drvdsmi(self):
        if self.drvdsmi is None:
            so_name = DSMI_SO_PATH
            self.drvdsmi = self.load_dll(so_name)
        return self.drvdsmi

    def get_drvhal_env_type(self):
        if self.drvhal is None:
            so_name = "libascend_hal.so"
            self.drvhal = self.load_dll(so_name)
        return self.drvhal

    def get_ascend_ml(self):
        if self.asendml is None and self.get_env_type() == "EP":
            dlog_path = _ascend_ml_library_path(ASCEND_TOOLKIT_ROOT, "libunified_dlog.so")
            ascend_ml_path = _ascend_ml_library_path(ASCEND_TOOLKIT_ROOT, "libascend_ml.so")
            if dlog_path is None or ascend_ml_path is None:
                missing = ", ".join(name for name, path in (
                    ("libunified_dlog.so", dlog_path), ("libascend_ml.so", ascend_ml_path)
                ) if path is None)
                sys.stderr.write(f"OSError: Ascend ML library not found under {ASCEND_TOOLKIT_ROOT}: {missing}")
                self.asendml = RetCode.FAILED
            else:
                self.load_dll(str(dlog_path))
                self.asendml = self.load_dll(str(ascend_ml_path))
        return self.asendml

    def get_env_type(self):
        if self.env_type:
            return self.env_type
        dev = self.get_drvhal_env_type()
        if dev != RetCode.FAILED:
            dev.drvGetPlatformInfo.argtypes = [ctypes.POINTER(ctypes.c_int)]
            num = ctypes.c_int(-1)
            ret = dev.drvGetPlatformInfo(ctypes.pointer(num))
            if ret == 0:
                if num.value == 0:
                    self.env_type = "RC"
                elif num.value == 1:
                    self.env_type = "EP"
        return self.env_type

    @staticmethod
    def ctypes_close_library(lib):
        if lib and lib != RetCode.FAILED:
            dlclose_func = ctypes.CDLL(None).dlclose
            dlclose_func.argtypes = [ctypes.c_void_p]
            dlclose_func.restype = ctypes.c_int
            dlclose_func(lib._handle)

    def dll_close(self):
        self.ctypes_close_library(self.drvdsmi)
        self.ctypes_close_library(self.drvhal)
        self.ctypes_close_library(self.asendml)
        self.ctypes_close_library(self.ascend_trace)
        self.drvdsmi = None
        self.drvhal = None
        self.asendml = None
        self.ascend_trace = None


class AmlAicoreInfo(ctypes.Structure):
    _fields_ = [
        ("aicCount", ctypes.c_uint16),
        ("aicVoltage", ctypes.c_uint16),
        ("aicFrequency", ctypes.c_uint16),
    ]


class _AscendMLLib:
    def __init__(self, ascendml):
        self.lib = ascendml
        fn = self.lib.AmlDeviceGetAicoreInfo
        fn.argtypes = [ctypes.c_int, ctypes.POINTER(AmlAicoreInfo)]
        fn.restype = ctypes.c_int

    def get_info(self, device):
        aic_info = AmlAicoreInfo()
        ret = self.lib.AmlDeviceGetAicoreInfo(int(device), ctypes.byref(aic_info))
        if ret != 0:
            return None, None, None
        return aic_info.aicCount, aic_info.aicVoltage, aic_info.aicFrequency


class Dsmi:
    def __init__(self, drvdsmi):
        self.lib = drvdsmi
        self.device_ids = self._get_device_ids()

    def _get_device_ids(self):
        count = ctypes.c_int(0)
        ret = self.lib.dsmi_get_device_count(ctypes.byref(count))
        if ret != 0:
            sys.exit(ret)

        device_array = (ctypes.c_int * count.value)()
        ret = self.lib.dsmi_list_device(device_array, count.value)
        if ret != 0:
            sys.exit(ret)

        return list(device_array)

    def get_aicore_rated_frequency(self, device_id):
        freq = ctypes.c_uint()
        ret = self.lib.dsmi_get_device_frequency(device_id, 9, ctypes.byref(freq))
        if ret != 0:
            sys.exit(ret)

        return freq.value

    def get_device_metrics(self, device_id):
        """Return instantaneous power, AI Core frequency, and voltage.

        Power comes from ``dsmi_get_device_power_info`` (unsigned short, 0.1 W).
        Frequency is ``curfreq`` from ``dsmi_get_aicore_info`` (MHz).
        Voltage comes from ``dsmi_get_device_voltage`` (0.01 V) and is returned
        in millivolts so it matches the existing AscendML ``aic_volt`` scale.
        """

        class DsmiPowerInfo(ctypes.Structure):
            _fields_ = [("power", ctypes.c_ushort)]

        class DsmiAicoreInfo(ctypes.Structure):
            _fields_ = [("freq", ctypes.c_uint), ("curfreq", ctypes.c_uint)]

        device_id = int(device_id)
        power_w = None
        power = DsmiPowerInfo()
        power_fn = self.lib.dsmi_get_device_power_info
        power_fn.argtypes = [ctypes.c_int, ctypes.POINTER(DsmiPowerInfo)]
        power_fn.restype = ctypes.c_int
        if power_fn(device_id, ctypes.byref(power)) == 0:
            power_w = power.power / 10.0

        frequency_mhz = None
        aicore = DsmiAicoreInfo()
        aicore_fn = self.lib.dsmi_get_aicore_info
        aicore_fn.argtypes = [ctypes.c_int, ctypes.POINTER(DsmiAicoreInfo)]
        aicore_fn.restype = ctypes.c_int
        if aicore_fn(device_id, ctypes.byref(aicore)) == 0:
            frequency_mhz = int(aicore.curfreq)

        volt_mv = None
        voltage = ctypes.c_uint()
        voltage_fn = self.lib.dsmi_get_device_voltage
        voltage_fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        voltage_fn.restype = ctypes.c_int
        if voltage_fn(device_id, ctypes.byref(voltage)) == 0:
            volt_mv = int(voltage.value) * 10

        if power_w is None and frequency_mhz is None and volt_mv is None:
            return None
        return {
            "power_w": power_w,
            "frequency_mhz": frequency_mhz,
            "volt_mv": volt_mv,
        }

    def set_aic_freq(self, device_id, freq):
        class SocStressCfg(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_char),
                ("set_restore", ctypes.c_char),
                ("value", ctypes.c_ushort),
            ]

        class LpmSocStressDsmiCfgIn(ctypes.Structure):
            _fields_ = [
                ("soc_stress_cfg", SocStressCfg),
                ("resv", ctypes.c_char * 32),
            ]

        buf = LpmSocStressDsmiCfgIn()
        buf.soc_stress_cfg.type = b"\x00"
        buf.soc_stress_cfg.set_restore = b"\x02"
        buf.soc_stress_cfg.value = freq

        ret = self.lib.dsmi_set_device_info(
            ctypes.c_uint(device_id),
            ctypes.c_uint(8),
            ctypes.c_uint(12),
            ctypes.byref(buf),
            ctypes.sizeof(buf),
        )

        if ret != 0:
            sys.stderr.write(f"Failed to set device info: error: {ret}")
            sys.exit(ret)

    def restore_aic_freq(self, device_id):
        rated_freq = self.get_aicore_rated_frequency(0)
        self.set_aic_freq(device_id, rated_freq)


class _AscendMLFacade:

    def get_info(self, device):
        lib = _loader.get_ascend_ml()
        if lib is None or lib == RetCode.FAILED:
            return {"device": int(device), "aic_freq": None, "aic_volt": None}
        aic_c, aic_v, aic_f = _AscendMLLib(lib).get_info(int(device))
        if aic_f is None and aic_v is None:
            return {"device": int(device), "aic_freq": None, "aic_volt": None}
        return {
            "device": int(device),
            "aic_count": int(aic_c) if aic_c is not None else None,
            "aic_freq": int(aic_f) if aic_f is not None else None,
            "aic_volt": int(aic_v) if aic_v is not None else None,
        }


_loader = LoadSoType()
_drvdsmi_lib = _loader.get_drvdsmi()
dsmi = Dsmi(_drvdsmi_lib) if _drvdsmi_lib != RetCode.FAILED else None
AscendML = _AscendMLFacade()
