using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn.Providers.Impl
{
    [SuppressUnmanagedCodeSecurity]
    [SecurityCritical]
    internal static class VclNativeMethods
    {
        internal const string ApiDllName = "vcltensormath";

        [DllImport(ApiDllName)]
        internal static extern int vclGetVersion();

        [DllImport(ApiDllName)]
        internal static extern void vclGemm(BlasTranspose transA, BlasTranspose transB,
                        int m, int n, int k,
                        double alpha, [In] double[] a, int aOffset,
                        [In] double[] b, int bOffset,
                        double beta, [In,Out] double[] c, int cOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclGemv(BlasTranspose transA, int m, int n,
                        double alpha, [In] double[] a, int aOffset, int aLength,
                        [In] double[] x, int xOffset, int xLength,
                        double beta, [In,Out] double[] y, int yOffset, int yLength);

        [DllImport(ApiDllName)]
        internal static extern void vclAxpy(int n, double alpha, [In] double[] x, int xOffset, [In,Out] double[] y, int yOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclAxpby(int n, double alpha, [In] double[] x, int xOffset, double beta, [In,Out] double[] y, int yOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclSet(int n, double alpha, [In,Out]double[] y, int yOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclCopy(int n, [In] double[] x, int xOffset, [In,Out] double[] y, int yOffset);
        
        [DllImport(ApiDllName)]
        internal static extern void vclAddScalar(int n, double alpha, [In,Out]double[] y, int yOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclAdd(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclSubstract(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclMultiply(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclDivide(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclPowx(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclSquare(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclExp(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclAbs(int n, [In] double[] a, int aOffset, [In,Out] double[] b, int bOffset);

        [DllImport(ApiDllName)]
        internal static extern double vclDot(int n, [In] double[] x, int xOffset, [In] double[] y, int yOffset);

        [DllImport(ApiDllName)]
        internal static extern double vclDotEx(int n, [In] double[] x, int xOffset, int incx, [In] double[] y, int yOffset, int incy);

        [DllImport(ApiDllName)]
        internal static extern double vclAsum(int n, [In] double[] x, int xOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclScaleInline(int n, double alpha, [In,Out]double[] x, int xOffset);

        [DllImport(ApiDllName)]
        internal static extern void vclScale(int n, double alpha, [In] double[] x, int xOffset, [In,Out] double[] y, int yOffset);
    }  
}
