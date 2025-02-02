from typing import Optional

import h5py


class HDF5:
    @staticmethod
    def set_nullable_attrs[T](key: str, value: Optional[T], attrs: h5py.AttributeManager):
        if value is None:
            attrs[key] = "None"
        else:
            attrs[key] = value

    @staticmethod
    def read_nullable_attrs(key: str, attrs: h5py.AttributeManager):
        if attrs[key] == "None":
            return None

        return attrs[key]
