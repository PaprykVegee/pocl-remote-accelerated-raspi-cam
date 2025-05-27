#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

float absdiff(__read_only image2d_t inputImageA, __read_only image2d_t inputImageB, int2 coord) {
    float4 pixel1 = read_imagef(inputImageA, imageSampler, coord);
    float4 pixel2 = read_imagef(inputImageB, imageSampler, coord);

    float gray1 = dot(pixel1.xyz, (float3)(0.2989f, 0.5870f, 0.1140f));
    float gray2 = dot(pixel2.xyz, (float3)(0.2989f, 0.5870f, 0.1140f));

    return fabs(gray1 - gray2);
}

float median(float* arr) {
    int n = KERNEL_WIDTH * KERNEL_HEIGHT;
    float temp;

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    int middle = n / 2;
    if (n % 2 == 0) {
        return (arr[middle - 1] + arr[middle]) / 2.0f;
    } else {
        return arr[middle];
    }
}

float erode(float* arr) {
    float min_val = arr[0];
    int n = KERNEL_WIDTH * KERNEL_HEIGHT;
    for (int i = 1; i < n; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

float dilate(float* arr) {
    float max_val = arr[0];
    for (int i = 1; i < KERNEL_WIDTH * KERNEL_HEIGHT; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}


// Kernel 1 – MEDIANA z różnicy dwóch obrazów
__kernel void median_kernel(__read_only image2d_t inputImageA,
                            __read_only image2d_t inputImageB,
                            __write_only image2d_t tempImage) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float arr[KERNEL_WIDTH * KERNEL_HEIGHT];
    int i = 0;

    int half_width = KERNEL_WIDTH / 2;
    int half_height = KERNEL_HEIGHT / 2;

    for (int dy = -half_height; dy <= half_height; dy++) {
        for (int dx = -half_width; dx <= half_width; dx++) {
            float diff = absdiff(inputImageA, inputImageB, coord + (int2)(dx, dy));
            float thres = step(0.06f, diff) * 255.0f;
            arr[i++] = thres;
        }
    }

    float med = median(arr);
    float normalized = med / 255.0f;
    write_imagef(tempImage, coord, (float4)(normalized, normalized, normalized, 1.0f));
}

// Kernel 2 – EROZJA
__kernel void erode_kernel(__read_only image2d_t tempImage,
                           __write_only image2d_t outputImage) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float arr[KERNEL_WIDTH * KERNEL_HEIGHT];
    int i = 0;

    int half_width = KERNEL_WIDTH / 2;
    int half_height = KERNEL_HEIGHT / 2;

    for (int dy = -half_height; dy <= half_height; dy++) {
        for (int dx = -half_width; dx <= half_width; dx++) {
            int2 sampleCoord = coord + (int2)(dx, dy);
            float4 pix = read_imagef(tempImage, imageSampler, sampleCoord);
            float gray = dot(pix.xyz, (float3)(0.2989f, 0.5870f, 0.1140f));
            arr[i++] = gray * 255.0f;
        }
    }

    float eroded = erode(arr);
    float normalized = eroded / 255.0f;
    write_imagef(outputImage, coord, (float4)(normalized, normalized, normalized, 1.0f));
}


// Kernel 3 - DYLATACJA
__kernel void dilate_kernel(__read_only image2d_t tempImage,
                            __write_only image2d_t outputImage) {
    
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float arr[KERNEL_WIDTH * KERNEL_HEIGHT];
    int i = 0;

    int half_width = KERNEL_WIDTH / 2;
    int half_height = KERNEL_HEIGHT / 2;

    for (int dy = -half_height; dy <= half_height; dy++) {
        for (int dx = -half_width; dx <= half_width; dx++) {
            int2 sampleCoord = coord + (int2)(dx, dy);
            float4 pix = read_imagef(tempImage, imageSampler, sampleCoord);
            float gray = dot(pix.xyz, (float3)(0.2989f, 0.5870f, 0.1140f));
            arr[i++] = gray * 255.0f;
        }
    }

    float result = dilate(arr);
    float normalized = result / 255.0f;
    write_imagef(outputImage, coord, (float4)(normalized, normalized, normalized, 1.0f));
}
