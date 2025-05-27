import pyopencl as cl
import numpy as np
import cv2
import time
from picamera2 import Picamera2

# Inicjalizacja kamery
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Pobranie pierwszej klatki
time.sleep(1)  # Czas na rozgrzanie kamery
prev_frame = picam2.capture_array()

if prev_frame is None:
    raise IOError("Nie udało się pobrać klatki z kamery")

prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
height, width, _ = prev_frame_rgb.shape

# OpenCL: Kontekst i kolejka
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# Format obrazu
fmt_rgba = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)

# Wczytanie kernela
with open("/home/toudi/Desktop/kernel.cl", 'r') as f:
    kernel_code = f.read()

program = cl.Program(ctx, kernel_code).build()

while True:
    start_time = time.time()

    # Pobranie nowej klatki
    curr_frame = picam2.capture_array()
    if curr_frame is None:
        print("Błąd pobierania klatki!")
        continue

    curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

    # Konwersja do RGBA
    prev_frame_rgba = np.dstack((prev_frame_rgb, np.full((height, width), 255, dtype=np.uint8)))
    curr_frame_rgba = np.dstack((curr_frame_rgb, np.full((height, width), 255, dtype=np.uint8)))

    # Utworzenie obrazów OpenCL
    input_imageA_cl = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               fmt_rgba, shape=(width, height), hostbuf=prev_frame_rgba)
    input_imageB_cl = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               fmt_rgba, shape=(width, height), hostbuf=curr_frame_rgba)

    temp_image_cl = cl.Image(ctx, cl.mem_flags.READ_WRITE, fmt_rgba, shape=(width, height))
    output_image_cl = cl.Image(ctx, cl.mem_flags.READ_WRITE, fmt_rgba, shape=(width, height))
    final_output_image_cl = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt_rgba, shape=(width, height))

    # Kernel 1: filtr medianowy
    median_kernel = program.median_kernel
    median_kernel.set_args(input_imageA_cl, input_imageB_cl, temp_image_cl)
    cl.enqueue_nd_range_kernel(queue, median_kernel, (width, height), None)

    # Kernel 2: erozja
    erode_kernel = program.erode_kernel
    erode_kernel.set_args(temp_image_cl, output_image_cl)
    cl.enqueue_nd_range_kernel(queue, erode_kernel, (width, height), None)

    # Kernel 3: dylatacja
    dilate_kernel = program.dilate_kernel
    dilate_kernel.set_args(output_image_cl, final_output_image_cl)
    cl.enqueue_nd_range_kernel(queue, dilate_kernel, (width, height), None)

    # Pobranie wyniku
    output_np = np.empty((height, width, 4), dtype=np.uint8)
    origin = (0, 0, 0)
    region = (width, height, 1)
    cl.enqueue_copy(queue, output_np, final_output_image_cl, origin=origin, region=region)
    queue.finish()

    # Konwersja do odcieni szarości
    gray_image = np.dot(output_np[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0

    # Wyświetlenie FPS na obrazie
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(gray_image, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)

    # Wyświetlenie wyniku
    cv2.imshow('Wynik', gray_image)

    # Aktualizacja klatki poprzedniej
    prev_frame_rgb = curr_frame_rgb.copy()

    # Wyjście po wciśnięciu ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Zwolnienie zasobów
picam2.close()
cv2.destroyAllWindows()
