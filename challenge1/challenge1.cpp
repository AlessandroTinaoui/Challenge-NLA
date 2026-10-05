#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>



#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

Eigen::SparseMatrix<double> buildConvMatrix(int rows, int cols, const Eigen::MatrixXd& H);
Eigen::MatrixXd addNoise(const Eigen::MatrixXd& img);
void saveMatrixAsPng(const std::string& filename, const Eigen::MatrixXd& M);
Eigen::VectorXd solveLinearSystemGMRES(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, double tol, int N);


int main(int argc, char** argv)
{   
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* input_image_path = argv[1];

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
            originalImg(i, j) = static_cast<double>(image_data[idx]); // non dovrebbero essere tra 0 e 255, dividendo per 255 rimarrebbero tra 0 e 1
        }
    }

    std::cout << "righe: " << height << "\ncolonne: " << width << "\n";


    // Task 2
    Eigen::MatrixXd noisyImg = addNoise(originalImg);
    saveMatrixAsPng("noisy_deer.png", noisyImg);

    // Task 3
    const int N = height * width;

    Eigen::VectorXd v(N);
    Eigen::VectorXd w(N);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            v(idx) = originalImg(i, j);
            w(idx) = noisyImg(i, j);
        }
    }

    std::cout << "Numero componenti atteso: " << N << "\n";
    std::cout << "v.size() = " << v.size() << "\n";
    std::cout << "w.size() = " << w.size() << "\n";
    std::cout << "Norma euclidea di v: ||v||_2 = " << v.norm() << "\n";


    // Task 4
    Eigen::SparseMatrix<double> A1 = buildConvMatrix(height, width, H_av1);

    std::cout << "Dimensioni di A1: " << A1.rows() << " x " << A1.cols() << "\n";
    std::cout << "Numero di entry non nulle in A1: " << A1.nonZeros() << "\n";

    // Task 5
    Eigen::VectorXd smoothed = A1 * w;
    Eigen::MatrixXd smoothedImg(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            smoothedImg(i, j) = smoothed(idx);
        }
    }

    saveMatrixAsPng("smoothed_noisy_deer.png", smoothedImg);


    // Task 6
    Eigen::SparseMatrix<double> A2 = buildConvMatrix(height, width, H_sh1);

    std::cout << "Dimensione di A2: " << A2.rows() << " x " << A2.cols() << "\n";
    std::cout << "Numero di entry non nulle in A2: " << A2.nonZeros() << "\n";

    // Verifica simmetria
    bool isSymmetricA2 = (A2 - Eigen::SparseMatrix<double>(A2.transpose())).norm() < 1e-12;
    std::cout << "A2 è simmetrica? " << (isSymmetricA2 ? "Sì" : "No") << "\n";

    // Task 7
    Eigen::VectorXd sharpened = A2 * v;

    Eigen::MatrixXd sharpenedImg(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            sharpenedImg(i, j) = sharpened(idx);
        }
    }

    saveMatrixAsPng("sharpened_deer.png", sharpenedImg);


    // Task 10
    Eigen::SparseMatrix<double> A3 = buildConvMatrix(height, width, H_ed2);

    std::cout << "Dimensione di A3: " << A3.rows() << " x " << A3.cols() << "\n";
    std::cout << "Numero di entry non nulle in A3: " << A3.nonZeros() << "\n";

    // Verifica simmetria
    bool isSymmetricA3 = (A3 - Eigen::SparseMatrix<double>(A3.transpose())).norm() < 1e-12;
    std::cout << "A3 è simmetrica? " << (isSymmetricA3 ? "Sì" : "No") << "\n";


    // Task 11
    Eigen::VectorXd edges = A3 * v;

    Eigen::MatrixXd edgesImg(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            edgesImg(i, j) = edges(idx);
        }
    }

    saveMatrixAsPng("edges_deer.png", edgesImg);


    // Task 12
    auto tol = 1e-10;
    Eigen::VectorXd y = solveLinearSystemGMRES(A3, w, tol, N);


    // Task 13
    Eigen::MatrixXd solution_y(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            solution_y(i, j) = y(idx);
        }
    }

    saveMatrixAsPng("solution_y.png", solution_y);

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

    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(rows, cols);

    out = M.unaryExpr([](double val) -> unsigned char {
        double clamped = std::max(0.0, std::min(255.0, val));
        return static_cast<unsigned char>(clamped);
    });

    stbi_write_png(filename.c_str(), cols, rows, 1, out.data(), cols);
}


Eigen::SparseMatrix<double> buildConvMatrix(int rows, int cols, const Eigen::MatrixXd& H){
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

                    if (H(k, l) != 0.0)
                        triplet.emplace_back(row, col, H(k, l));
                }
            }
        }
    }

    A.setFromTriplets(triplet.begin(), triplet.end());
    
    return A;
}

Eigen::VectorXd solveLinearSystemGMRES(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, double tol, int N){
    Eigen::SparseMatrix<double> I(N, N);
    I.setIdentity();
    Eigen::SparseMatrix<double>  A4 = A + 4.0 * I;

    // Verifica simmetria
    bool isSymmetricA4 = (A4 - Eigen::SparseMatrix<double>(A4.transpose())).norm() < 1e-12;
    std::cout << "A4 è simmetrica? " << (isSymmetricA4 ? "Sì" : "No") << "\n";
    
    Eigen::VectorXd x;
    Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> GMRES;    
    GMRES.setTolerance(tol);
    GMRES.compute(A4);
    if(GMRES.info() == Eigen::Success){
        x = GMRES.solve(b);
        std::cout << "#iterations:     " << GMRES.iterations() << std::endl;
        std::cout << "relative residual: " << GMRES.error()      << std::endl;
    }
    else{
        std::cout << "ATTENZIONE: Il solutore ha fallito " << GMRES.info() << "\n";
    }

    return x;
}