#include <iostream>
#include <fstream>
#include <cstdio>
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

void saveMatrixAsPng(const std::string& filename, const Eigen::MatrixXd& M);
Eigen::MatrixXd addNoise(const Eigen::MatrixXd& img);


int main(int argc, char** argv)
{   
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* input_image_path = argv[1];
 
    int width = 0, height = 0, channels = 0;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Errore: impossibile caricare l'immagine " << input_image_path << "\n";
        return 1;
    }

    Eigen::MatrixXd A(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            A(i, j) = static_cast<double>(image_data[idx]); 
        }
    }

    //Task 1
    Eigen::MatrixXd AtA = A.transpose() * A;
    std::cout << "Euclidean norm: " << AtA.norm() << std::endl;

    stbi_image_free(image_data);

    //Task 2
    bool isSymmetricAtA = (AtA - Eigen::MatrixXd(AtA.transpose())).norm() < 1e-12;
    std::cout << "At*A è simmetrica? " << (isSymmetricAtA ? "Sì" : "No") << "\n";

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(AtA, Eigen::EigenvaluesOnly);
    if (eigensolver.info() != Eigen::Success) abort();
    Eigen::VectorXd lambda = eigensolver.eigenvalues(); // ordine crescente
    int n = lambda.size();
    std::cout << "Largest eigenvalue of A^T A: " << lambda(n - 1) << "\n";
    std::cout << "sigma_1 = " << std::sqrt(lambda(n - 1)) << "\n";
    std::cout << "sigma_2 = " << std::sqrt(lambda(n - 2)) << std::endl;

    //Task 3 e 4
    if (!Eigen::saveMarket(AtA, "AtA.mtx")) {
        std::cerr << "Errore: impossibile esportare AtA.mtx\n";
        return 1;
    }

    //Task 5
    Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd W = svd.singularValues();
    std::cout << "Norm of W: " << W.norm() << std::endl;

    //Task 6
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    int k1 = 30;
    Eigen::MatrixXd C1 = U.leftCols(k1);
    Eigen::MatrixXd D1 = V.leftCols(k1) * W.head(k1).asDiagonal();
 
    std::cout << "Numero di entry non nulle in C1: " << C1.nonZeros() << std::endl;
    std::cout << "Numero di entry non nulle in D1: " << D1.nonZeros() << std::endl;

    int k2 = 90;
    Eigen::MatrixXd C2 = U.leftCols(k2);
    Eigen::MatrixXd D2 = V.leftCols(k2) * W.head(k2).asDiagonal();
 
    std::cout << "Numero di entry non nulle in C2: " << C2.nonZeros() << std::endl;
    std::cout << "Numero di entry non nulle in D2: " << D2.nonZeros() << std::endl;

    //Task 7
    Eigen::MatrixXd A_compressed1, A_compressed2;
    A_compressed1 = C1 * D1.transpose();
    saveMatrixAsPng("A_compressed1.png", A_compressed1);
    A_compressed2 = C2 * D2.transpose();
    saveMatrixAsPng("A_compressed2.png", A_compressed2);

    // Task 8
    int dim = 400;
    Eigen::MatrixXd chessboard(dim,dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            chessboard(i, j) = ((i/40 + j/40)%2 == 0) ? 0.0 : 255.0; 
        }
    }

    saveMatrixAsPng("chessboard.png", chessboard);
    std::cout << "euclidean norm: " << chessboard.norm() << std::endl;

    //Task 9
    Eigen::MatrixXd noisyChessboard = addNoise(chessboard);
    saveMatrixAsPng("noisyChessboard.png", noisyChessboard);


    return 0;
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

Eigen::MatrixXd addNoise(const Eigen::MatrixXd& img)
{
    Eigen::MatrixXd noise = 60.0 * Eigen::MatrixXd::Random(img.rows(), img.cols());
    return (img + noise).cwiseMax(0.0).cwiseMin(255.0);
}