from dataclasses import dataclass
import json


@dataclass
class WorkerInfo:
    ip: str
    port: int
    queue_size: int = 0
    id: int = 0

    @staticmethod
    def from_json(json_str):
        return WorkerInfo(**json.loads(json_str))


class Sendable:

    def _as_bytes(self, value):
        # encode to binary and add length prefix
        dtype = type(value)
        b = None
        if dtype == str:
            b = value.encode('utf-8')
        elif dtype == int:
            b = value.to_bytes(4, 'big')
        elif dtype == float:
            b = value.to_bytes(8, 'big')
        elif dtype == list:
            b = b''
            for item in value:
                b += self._as_bytes(item)
        elif dtype == bytes:
            b = value

        # add length prefix
        return len(b).to_bytes(4, 'big') + b
    
    def to_binary(self):
        values = vars(self)

        value_bytes = [ self._as_bytes(value) for value in values.values() ]
        return b''.join(value_bytes)
    
    def _from_bytes(self, b, dtype):
        if dtype == str:
            return b.decode('utf-8')
        elif dtype == int:
            return int.from_bytes(b, 'big')
        elif dtype == float:
            return int.from_bytes(b, 'big')
        elif dtype == list:
            values = []
            i = 0
            while i < len(b):
                length = int.from_bytes(b[i:i+4], 'big')
                values.append(self._from_bytes(b[i+4:i+4+length], str))
                i += 4 + length
            return values
        elif dtype == bytes:
            return b
    
    def load_binary(self, b):
        values = vars(self)
        i = 0
        for key, value in values.items():
            length = int.from_bytes(b[i:i+4], 'big')
            setattr(self, key, self._from_bytes(b[i+4:i+4+length], type(value)))
            i += 4 + length


@dataclass
class TestClass(Sendable):
    a: str
    b: int
    c: float
    d: list[str]
    e: bytes


def _test():
    x = TestClass("hello", 42, 3.14, ["a", "b", "c"], b"hello")
    print(x)
    b = x.to_binary()
    print(len(b))

    y = TestClass()
    y.load_binary(b)
    print(y)

if __name__ == "__main__":
    _test()

