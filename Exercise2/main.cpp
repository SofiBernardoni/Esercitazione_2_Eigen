#include <iostream>
#include "Eigen/Eigen"
#include <math.h>
#include <iomanip>

using namespace std;
using namespace Eigen;

template<typename MatrixType, typename VectorType>
void solveLinearSystem(const MatrixType& A, const VectorType& b, int i)
{
    // Eseguire la fattorizzazione LU
    PartialPivLU<MatrixXd> lu_fatt(A);

    // Risolvo con PALU
    VectorXd x1 = lu_fatt.solve(b);

    // Fattorizzazione QR
    HouseholderQR<MatrixXd> qr_fatt(A);

    // risolvo con QR
    VectorXd x2=qr_fatt.solve(b);

   // vettore soluzione
    Vector2d v=Vector2d::Zero(2);
    v << -1.0e+0, -1.0e+00;
    double nv=v.norm();

    Vector2d errlu=x1-v;
    Vector2d errqr=x2-v;

    double nelu=(errlu.norm())/nv;
    double neqr=(errqr.norm())/nv;

    cout << "L'errore relativo della matrice A" << i << " con la fattorizzazione PALU è: "<< endl << nelu << setprecision(8) << scientific <<
        endl <<" e con la fattorizzazione QR: " << endl << neqr << endl;
}

int main()
{
    Matrix2d A1=Matrix2d::Zero(2,2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;

    Vector2d b1=Vector2d::Zero(2); // vettore colonna
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    Matrix2d A2=Matrix2d::Zero(2,2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01,-8.324762492991313e-01;

    Vector2d b2=Vector2d::Zero(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    Matrix2d A3=Matrix2d::Zero(2,2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b3=Vector2d::Zero(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    solveLinearSystem(A1, b1, 1);
    solveLinearSystem(A2, b2, 2);
    solveLinearSystem(A3, b3, 3);

    return 0;
}

/**Matrici P, L e U
    MatrixXd P = lu.permutationP();
    MatrixXd L = lu.matrixLU().triangularView<Lower>();
    MatrixXd U = lu.matrixLU().triangularView<Upper>():**/

/**Matrici Q e R
    MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixQR().triangularView<Upper>();**/

