#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "bmp.h"



const int image_width = 9984;
const int image_height = 5616;
const int image_size = image_width*image_height;
const int color_depth = 255;

// method to write data to the bmp file
void write_bmp(unsigned char* data, int width, int height){
    struct bmp_id id;
    id.magic1 = 0x42;
    id.magic2 = 0x4D;

    struct bmp_header header;
    header.file_size = width*height+54 + 2;
    header.pixel_offset = 1078;

    struct bmp_dib_header dib_header;
    dib_header.header_size = 40;
    dib_header.width = width;
    dib_header.height = height;
    dib_header.num_planes = 1;
    dib_header.bit_pr_pixel = 8;
    dib_header.compress_type = 0;
    dib_header.data_size = width*height;
    dib_header.hres = 0;
    dib_header.vres = 0;
    dib_header.num_colors = 256;
    dib_header.num_imp_colors = 0;

    char padding[2];

    unsigned char* color_table = (unsigned char*)malloc(1024);
    for(int c= 0; c < 256; c++){
        color_table[c*4] = (unsigned char) c;
        color_table[c*4+1] = (unsigned char) c;
        color_table[c*4+2] = (unsigned char) c;
        color_table[c*4+3] = 0;
    }

    // writing header and image data into the bmp image file
    FILE* fp = fopen("out.bmp", "w+");
    fwrite((void*)&id, 1, 2, fp);
    fwrite((void*)&header, 1, 12, fp);
    fwrite((void*)&dib_header, 1, 40, fp);
    fwrite((void*)color_table, 1, 1024, fp);
    fwrite((void*)data, 1, width*height, fp);
    fwrite((void*)&padding,1,2,fp);
    fclose(fp);
}

//method to read data from a bmp image file
unsigned char* read_bmp(char* filename){

    FILE* fp = fopen(filename, "rb");

    int width, height, offset;

    // read height, width and offset from the file
    fseek(fp, 18, SEEK_SET);
    fread(&width, 4, 1, fp);
    fseek(fp, 22, SEEK_SET);
    fread(&height, 4, 1, fp);
    fseek(fp, 10, SEEK_SET);
    fread(&offset, 4, 1, fp);

    // read data from the file
    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char)*height*width);

    fseek(fp, offset, SEEK_SET);
    
    // ignore the padding
    fread(data, sizeof(unsigned char), height*width, fp);

    fclose(fp);

    return data;
}


int main(int argc, char** argv){

    int n_threads = omp_get_max_threads();

    unsigned char* image = read_bmp("foggy_1.bmp");
    unsigned char* output_image = (unsigned char*)malloc(sizeof(unsigned char) * image_size);
    int* histogram = (int*)calloc(sizeof(int), color_depth);

#pragma omp parallel for num_threads(n_threads)
    for(int i = 0; i < image_size; i++){
        int image_val = image[i]; 
        #pragma omp critical
                histogram[image_val]++;
        }


    float* transfer_function = (float*)calloc(sizeof(float), color_depth);
    
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for(int i = 0; i < color_depth; i++){
        for(int j = 0; j < i+1; j++){
            transfer_function[i] += color_depth*((float)histogram[j])/(image_size);
        }
    }


#pragma omp parallel for num_threads(n_threads)
    for(int i = 0; i < image_size; i++){
        output_image[i] = transfer_function[image[i]];
    }

    write_bmp(output_image, image_width, image_height);
}