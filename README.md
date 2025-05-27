# Setting Up `pocl-remote` Server and Client (Ubuntu 24.04)

Official documentation:  
[https://portablecl.org/docs/html/remote.html](https://portablecl.org/docs/html/remote.html)

---

## Project Goal

The goal of this project was to **run `pocl-remote` on a Raspberry Pi** device equipped with a camera, where:
- The Raspberry Pi captures image data and sends it for processing,
- OpenCL computations are **offloaded to a remote external GPU** (e.g., a workstation),
- With `pocl-remote`, the lightweight client (Raspberry Pi) delegates the compute load to a more powerful GPU server.

---

## Note

The official method for building the `pocld` server may fail with an error like:

```
   42 | #include "config.h"
      |          ^~~~~~~~~~
```

Below is a verified working method to build the server and client for `pocl-remote`, tested on **Ubuntu 24.04**.

---

## Building the `pocl-r` Server (on the host machine)

### 1. Clone the repository

```bash
git clone https://github.com/pocl/pocl.git
cd pocl
```

### 2. Navigate to the server source directory

```bash
cd pocld
```

### 3. Create and enter the `build` directory

```bash
mkdir build
cd build
```

### 4. Set OpenCL version

In the `CMakeLists.txt` file located in the `pocld` directory, add:

```cmake
set(OPENCL_HEADER_VERSION 300)
```

### 5. Configure the project

```bash
cmake -DCMAKE_C_FLAGS="-DHAVE_CLOCK_GETTIME=1 -DHAVE_FORK=1" ../
```

### 6. Rename the misnamed config file

```bash
mv pocld_config.h config.h
```

### 7. Edit the `pocl_intfn.h` file

In the file:

```
~/pocl/lib/CL/pocl_intfn.h
```

Comment out the following line:

```cpp
// POdeclsym(clSetCommandQueueProperty)
```

### 8. Build the server

```bash
make
```

---

## 📱 Building the `pocl-remote` Client (on the target device, e.g. Raspberry Pi)

### 1. Clone the repository

```bash
git clone https://github.com/pocl/pocl.git
cd pocl
```

### 2. Build the client

```bash
mkdir build
cd build
cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_LOADABLE_DRIVERS=0 -DENABLE_ICD=1 -DENABLE_REMOTE_CLIENT=1 ..
make -j$(nproc)
```

---

## Client Configuration

Before launching the server, configure the client environment:

```bash
export OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd
export POCL_DEVICES=remote
export POCL_REMOTE0_PARAMETERS='<IP_ADDRESS>:<PORT>'
```

Example:

```bash
export POCL_REMOTE0_PARAMETERS='192.168.1.100:5678'
```

---

## Launching the Server

Run the server with:

```bash
./pocld -a <IP_ADDRESS> -p <PORT>
```

Example:

```bash
./pocld -a 192.168.1.100 -p 5678
```

> **Note:** The server IP should start with `192...`. Other IPs may fail to connect.

---

## Summary

- The `pocld` server must be built manually with some adjustments.
- The client can be built using the official documentation.
- Always set the environment variables on the client **before starting the server**.
- This setup allows **lightweight devices like Raspberry Pi** to use powerful remote GPUs for OpenCL computation.
