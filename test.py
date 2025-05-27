import pyopencl as cl
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

imageA = np.array(Image.open("/home/plorenc/Desktop/AiR_ISS/AVS/pedestrian/input/in000519.jpg").convert("RGB"), dtype=np.uint8)
imageB = np.array(Image.open("/home/plorenc/Desktop/AiR_ISS/AVS/pedestrian/input/in000520.jpg").convert("RGB"), dtype=np.uint8)

# Opcjonalnie zmniejsz obrazy, jeśli są za duże
max_dim = 128  # Zmienna określająca maksymalny rozmiar obrazu
if imageA.shape[0] > max_dim or imageA.shape[1] > max_dim:
    aspect_ratio = imageA.shape[1] / imageA.shape[0]
    new_height = max_dim
    new_width = int(new_height * aspect_ratio)
    imageA = np.array(Image.fromarray(imageA).resize((new_width, new_height)), dtype=np.uint8)

if imageB.shape[0] > max_dim or imageB.shape[1] > max_dim:
    aspect_ratio = imageB.shape[1] / imageB.shape[0]
    new_height = max_dim
    new_width = int(new_height * aspect_ratio)
    imageB = np.array(Image.fromarray(imageB).resize((new_width, new_height)), dtype=np.uint8)

# Inicjalizacja OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Przekształcenie obrazów do odpowiedniego formatu dla OpenCL
imageA_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imageA)
imageB_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imageB)

# Tworzenie pustego bufora na wynik
output_image = np.zeros_like(imageA)
output_image_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output_image.nbytes)

# Kernel OpenCL
kernel_code = """
__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

float absdiff(__read_only image2d_t inputImageA, __read_only image2d_t inputImageB, int2 coord) {
    float4 pixel1 = convert_float4(read_imageui(inputImageA, imageSampler, coord));
    float4 pixel2 = convert_float4(read_imageui(inputImageB, imageSampler, coord));

    float3 rgb1 = convert_float3(pixel1.xyz);
    float3 rgb2 = convert_float3(pixel2.xyz);

    float gray1 = dot(rgb1, (float3)(0.2989f, 0.5870f, 0.1140f));
    float gray2 = dot(rgb2, (float3)(0.2989f, 0.5870f, 0.1140f));

    return fabs(gray1 - gray2);
}

__kernel void absdidsadaff(__read_only image2d_t inputImageA, __read_only image2d_t inputImageB, __write_only image2d_t outputImage) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    // Oblicz różnicę
    float diff = absdiff(inputImageA, inputImageB, coord);

    // Zapisz wynik
    uint4 result = (uint4)(diff, diff, diff, 255);  // cała różnica w jednym kanale
    write_imageui(outputImage, coord, result);
}
"""

# Kompilacja kernela
program = cl.Program(context, kernel_code).build()

# Tworzenie obrazów OpenCL z użyciem create_image
imageA_2d = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.RGB, cl.channel_type.UNSIGNED_INT8), shape=(imageA.shape[1], imageA.shape[0]))
imageB_2d = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.RGB, cl.channel_type.UNSIGNED_INT8), shape=(imageB.shape[1], imageB.shape[0]))

# Kopiowanie danych do obrazów OpenCL (Użyj create_image do inicjalizacji)
cl.enqueue_copy(queue, imageA_2d, imageA, origin=(0, 0), region=(imageA.shape[1], imageA.shape[0]))  # Szerokość, Wysokość
cl.enqueue_copy(queue, imageB_2d, imageB, origin=(0, 0), region=(imageB.shape[1], imageB.shape[0]))  # Szerokość, Wysokość

# Wykonanie kernela
program.absdidsadaff(queue, imageA.shape[::-1], None, imageA_2d, imageB_2d, output_image_buf)

# Zbieranie wyników
cl.enqueue_copy(queue, output_image, output_image_buf).wait()

# Zapisanie wynikowego obrazu
output_image_pil = Image.fromarray(output_image)
output_image_pil.save("/home/plorenc/Desktop/absdiff_result.png")

# Wyświetlenie wynikowego obrazu
plt.imshow(output_image)
plt.show()
