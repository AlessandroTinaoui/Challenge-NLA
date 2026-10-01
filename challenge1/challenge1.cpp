#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

Eigen::SparseMatrix<double> buildConvMatrix(int rows, int cols, const Eigen::MatrixXd& H);
Eigen::MatrixXd addNoise(const Eigen::MatrixXd& img);
void saveMatrixAsPng(const std::string& filename, const Eigen::MatrixXd& M);

int main(int argc, char** argv)
{   
    // conv kernels
    Eigen::MatrixXd H_av1(3, 3);
    H_av1 << 1.0, 1.0, 1.0,
             1.0, 4.0, 1.0,
             1.0, 1.0, 1.0;
    H_av1 /= 12.0;

    Eigen::MatrixXd H_sh1(3, 3);
    H_sh1 <<  0.0, -3.0,  0.0,
             -1.0,  9.0, -3.0,
              0.0, -1.0,  0.0;

    Eigen::MatrixXd H_ed2(3, 3);
    H_ed2 << -1.0, 0.0, 1.0,
             -2.0, 0.0, 2.0,
             -1.0, 0.0, 1.0;
    

    

    // Task 1 
    const char* input_image_path = "challenge1/deer.jpg";
    int width = 0, height = 0, channels = 0;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Errore: impossibile caricare l'immagine " << input_image_path << "\n";
        return 1;
    }

    Eigen::MatrixXd originalImg(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            // originalImg(i, j) = static_cast<double>(image_data[idx]) / 255.0;
            originalImg(i, j) = static_cast<double>(image_data[idx]); // non dovrebbero essere tra 0 e 255, dividendo per 255 rimarrebbero tra 0 e 1
        }
    }

    std::cout << "righe: " << height << "\ncolonne: " << width << "\n";


    // Task 2
    Eigen::MatrixXd noisyImg = addNoise(originalImg);
    saveMatrixAsPng("noisy_deer.png", noisyImg);


    return 0;
}


Eigen::MatrixXd addNoise(const Eigen::MatrixXd& img)
{
    Eigen::MatrixXd noise = 50.0 * Eigen::MatrixXd::Random(img.rows(), img.cols());
    return img + noise;
}


void saveMatrixAsPng(const std::string& filename, const Eigen::MatrixXd& M)
{
    const int rows = static_cast<int>(M.rows());
    const int cols = static_cast<int>(M.cols());

    std::vector<unsigned char> buffer(rows * cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const int idx = i * cols + j;

            // Clamp tra 0 e 255 per evitare overflow nella conversione a unsigned char
            double val = std::max(0.0, std::min(255.0, M(i, j)));
            buffer[idx] = static_cast<unsigned char>(val);
        }
    }

    stbi_write_png(filename.c_str(), cols, rows, 1, buffer.data(), cols);
}


Eigen::SparseMatrix<double> buildConvMatrix(
    int rows, int cols, const Eigen::MatrixXd& H){
    const int kh = H.rows();     // kernel height
    const int kw = H.cols();     // kernel width 
    const int ci = kh / 2;       // center offset, rows  -> (n-1)/2
    const int cj = kw / 2;       // center offset, cols  -> (m-1)/2
    const int N = rows * cols;   // convolution matrix size

    Eigen::SparseMatrix<double> A(N, N);
    std::vector<Eigen::Triplet<double>> triplet;
    triplet.reserve(static_cast<size_t>(N) * static_cast<size_t>(H.size())); 

    // image loop
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            // row of A
            const int row = i * cols + j;

            // index of upper left corner of the kernel
            const int i_base = i - ci;
            const int j_base = j - cj;    

            // kernel loop
            for (int k = 0; k < kh; ++k) {  
                for (int l = 0; l < kw; ++l) {
                    const int ii = i_base + k;
                    const int jj = j_base + l;
                    
                    // if out of bounds, skip
                    if (ii < 0 || ii >= rows || jj < 0 || jj >= cols)
                        continue;

                    const int col = ii * cols + jj;

                    triplet.emplace_back(row, col, H(k, l));
                }
            }
        }
    }

    A.setFromTriplets(triplet.begin(), triplet.end());
    
    return A;
}