To compile the C++ samples for use in Python with `ctypes` use the following commands:
```
g++ -c -fPIC qtvmc.cpp
g++ -shared -o qtvmclib.so qtvmc.o
```
and the same for `spacevmc.cpp`. The first line will create the `qtvmc.o` file and the second line will create the actual library `qtvmclib.so` that is loaded by `ctypes`.

This process has to be repeated when using a new computer.
