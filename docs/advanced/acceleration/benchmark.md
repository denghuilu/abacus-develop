# Benchmarks

## si16_pw

### test configurations
**INPUT**
```
INPUT_PARAMETERS
#Parameters (1.General)
suffix                  autotest
calculation             scf
ntype                   1
#nbands                 8
symmetry                1
device                  gpu

#Parameters (2.Iteration)
ecutwfc                 60
scf_thr                 1e-8
scf_nmax                100
cal_force               1
cal_stress              1
#Parameters (3.Basis)
basis_type              pw

#Parameters (4.Smearing)
smearing_method         gauss
smearing_sigma          0.002

#Parameters (5.Mixing)
mixing_type             pulay
mixing_beta             0.3
```
**KPT**
```
K_POINTS
0
Gamma
5 5 5 0 0 0
```
**STRU**
```
ATOMIC_SPECIES
Si 14 ../../../tests/PP_ORB/Si_ONCV_PBE-1.0.upfq

LATTICE_CONSTANT
1

LATTICE_VECTORS
 10.2000000  10.2000000   0.0000000
 10.2000000   0.0000000  10.200[]()0000
  0.0000000  10.2000000  10.2000000

ATOMIC_POSITIONS
Direct

Si
0.0
16
  0.0000000   0.0000000   0.0000000 1 1 1
  0.1250000   0.1250000   0.1250000 1 1 1
  0.0000000   0.0000000   0.5000000 1 1 1
  0.1250000   0.1250000   0.6250000 1 1 1
  0.0000000   0.5000000   0.0000000 1 1 1
  0.1250000   0.6250000   0.1250000 1 1 1
  0.0000000   0.5000000   0.5000000 1 1 1
  0.1250000   0.6250000   0.6250000 1 1 1
  0.5000000   0.0000000   0.0000000 1 1 1
  0.6250000   0.1250000   0.1250000 1 1 1
  0.5000000   0.0000000   0.5000000 1 1 1
  0.6250000   0.1250000   0.6250000 1 1 1
  0.5000000   0.5000000   0.0000000 1 1 1
  0.6250000   0.6250000   0.1250000 1 1 1
  0.5000000   0.5000000   0.5000000 1 1 1
  0.6250000   0.6250000   0.6250000 1 1 1

```
### device conigurations
- 2 * 3090 GPU
- 2(core) * Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz

### CPU results:
```
 *********************************************************
 *                                                       *
 *                  WELCOME TO ABACUS v3.0               *
 *                                                       *
 *            'Atomic-orbital Based Ab-initio            *
 *                  Computation at UStc'                 *
 *                                                       *
 *          Website: http://abacus.ustc.edu.cn/          *
 *                                                       *
 *********************************************************
 Sat Nov 19 22:12:05 2022
 MAKE THE DIR         : OUT.autotest/
 UNIFORM GRID DIM     : 72 * 72 * 72
 UNIFORM GRID DIM(BIG): 72 * 72 * 72
 DONE(0.638032   SEC) : SETUP UNITCELL
 DONE(0.669173   SEC) : SYMMETRY
 DONE(0.6701     SEC) : INIT K-POINTS
 ---------------------------------------------------------
 Self-consistent calculations for electrons
 ---------------------------------------------------------
 SPIN    KPOINTS         PROCESSORS  
 1       10              2           
 ---------------------------------------------------------
 Use plane wave basis
 ---------------------------------------------------------
 ELEMENT NATOM       XC          
 Si      16          
 ---------------------------------------------------------
 Initial plane wave basis and FFT box
 ---------------------------------------------------------
 DONE(0.688293   SEC) : INIT PLANEWAVE
 MEMORY FOR PSI (MB)  : 53.4357
 DONE(0.750558   SEC) : LOCAL POTENTIAL
 DONE(0.776818   SEC) : NON-LOCAL POTENTIAL
 DONE(0.7777     SEC) : INIT BASIS
 -------------------------------------------
 STEP OF RELAXATION : 1
 -------------------------------------------
 START CHARGE      : atomic
 DONE(0.98475    SEC) : INIT SCF
 ITER   ETOT(eV)       EDIFF(eV)      DRHO       TIME(s)    
 CG1    -1.715993e+03  0.000000e+00   4.410e-01  8.873e+01  
 CG2    -1.715532e+03  4.612549e-01   1.895e-01  1.265e+01  
 CG3    -1.715732e+03  -1.998605e-01  4.381e-03  1.404e+01  
 CG4    -1.715743e+03  -1.069036e-02  4.033e-03  2.377e+01  
 CG5    -1.715750e+03  -7.109062e-03  5.397e-05  1.476e+01  
 CG6    -1.715750e+03  -2.715578e-04  5.044e-05  2.693e+01  
 CG7    -1.715750e+03  -1.857458e-04  2.426e-05  1.310e+01  
 CG8    -1.715750e+03  -1.352547e-04  7.944e-05  1.286e+01  
 CG9    -1.715750e+03  6.856684e-04   1.858e-04  1.492e+01  
 CG10   -1.715750e+03  -3.129368e-04  1.255e-05  1.912e+01  
 CG11   -1.715750e+03  1.742670e-06   3.886e-05  1.630e+01  
 CG12   -1.715750e+03  -1.139976e-05  8.168e-06  1.477e+01  
 CG13   -1.715750e+03  -5.350675e-06  9.103e-07  1.270e+01  
 CG14   -1.715750e+03  -3.386135e-06  2.056e-07  2.122e+01  
 CG15   -1.715750e+03  2.312896e-07   1.013e-07  1.430e+01  
 CG16   -1.715750e+03  -6.495933e-08  1.603e-08  1.460e+01  
 CG17   -1.715750e+03  -2.460690e-08  1.429e-09  1.886e+01  
 ><><><><><><><><><><><><><><><><><><><><><><
 TOTAL-STRESS (KBAR):
 ><><><><><><><><><><><><><><><><><><><><><><
 4.127e+01      1.196e-14      -1.196e-14     
 0.000e+00      4.127e+01      -3.987e-15     
 0.000e+00      -3.987e-15     4.127e+01      
 TOTAL-PRESSURE: 4.127e+01 KBAR

  |CLASS_NAME---------|NAME---------------|TIME(Sec)-----|CALLS----|AVG------|PER%-------
                       total               358.4          17        21        1e+02     %
   Driver              driver_line         358.34         1         3.6e+02   1e+02     %
   PW_Basis            setuptransform      0.56678        1         0.57      0.16      %
   ESolver_KS_PW       Run                 353.84         1         3.5e+02   99        %
   PW_Basis            recip2real          0.45072        99        0.0046    0.13      %
   PW_Basis            gathers_scatterp    0.13226        99        0.0013    0.037     %
   Potential           init_pot            0.12502        1         0.13      0.035     %
   Potential           update_from_charge  2.0993         18        0.12      0.59      %
   Potential           cal_v_eff           2.0936         18        0.12      0.58      %
   H_Hartree_pw        v_hartree           0.20738        18        0.012     0.058     %
   PW_Basis            real2recip          0.73765        156       0.0047    0.21      %
   PW_Basis            gatherp_scatters    0.18509        156       0.0012    0.052     %
   PotXC               cal_v_eff           1.873          18        0.1       0.52      %
   XC_Functional       v_xc                1.9689         19        0.1       0.55      %
   Symmetry            rho_symmetry        0.20729        19        0.011     0.058     %
   HSolverPW           solve               350.86         18        19        98        %
   Nonlocal            getvnl              1.3218         180       0.0073    0.37      %
   pp_cell_vnl         getvnl              1.4616         200       0.0073    0.41      %
   WF_igk              get_sk              0.63908        4000      0.00016   0.18      %
   DiagoIterAssist     diagH_subspace      42.748         170       0.25      12        %
   OperatorPW          hPsi                299.2          26572     0.011     83        %
   Operator            EkineticPW          0.68335        26572     2.6e-05   0.19      %
   Operator            VeffPW              188.16         26572     0.0071    53        %
   PW_Basis_K          recip2real          105.78         39303     0.0027    30        %
   PW_Basis_K          gathers_scatterp    19.217         39303     0.00049   5.4       %
   PW_Basis_K          real2recip          81.739         33542     0.0024    23        %
   PW_Basis_K          gatherp_scatters    12.947         33542     0.00039   3.6       %
   Operator            NonlocalPW          109.87         26572     0.0041    31        %
   Nonlocal            add_nonlocal_pp     52.356         26572     0.002     15        %
   DiagoIterAssist     LAPACK_subspace     0.11608        170       0.00068   0.032     %
   DiagoCG             diag_once           289.6          180       1.6       81        %
   ElecStatePW         psiToRho            15.968         18        0.89      4.5       %
   Charge              rho_mpi             0.16964        18        0.0094    0.047     %
   Charge              mix_rho             0.30295        16        0.019     0.085     %
   Forces              cal_force_nl        0.56564        1         0.57      0.16      %
   Stress_PW           cal_stress          2.9824         1         3         0.83      %
   Stress_Func         stres_nl            2.8122         1         2.8       0.78      %
 ----------------------------------------------------------------------------------------

 START  Time  : Sat Nov 19 22:12:05 2022
 FINISH Time  : Sat Nov 19 22:18:03 2022
 TOTAL  Time  : 3.6e+02
 SEE INFORMATION IN : OUT.autotest/
[WARNING] yaksa: 1 leaked handle pool objects
[WARNING] yaksa: 1 leaked handle pool objects

Process finished with exit code 0
```

### GPU results
```
 *********************************************************
 *                                                       *
 *                  WELCOME TO ABACUS v3.0               *
 *                                                       *
 *            'Atomic-orbital Based Ab-initio            *
 *                  Computation at UStc'                 *
 *                                                       *
 *          Website: http://abacus.ustc.edu.cn/          *
 *                                                       *
 *********************************************************
 Sat Nov 19 22:10:19 2022
 MAKE THE DIR         : OUT.autotest/
 UNIFORM GRID DIM     : 72 * 72 * 72
 UNIFORM GRID DIM(BIG): 72 * 72 * 72
 DONE(1.90608    SEC) : SETUP UNITCELL
 DONE(1.93575    SEC) : SYMMETRY
 DONE(1.93665    SEC) : INIT K-POINTS
 ---------------------------------------------------------
 Self-consistent calculations for electrons
 ---------------------------------------------------------
 SPIN    KPOINTS         PROCESSORS  
 1       10              2           
 ---------------------------------------------------------
 Use plane wave basis
 ---------------------------------------------------------
 ELEMENT NATOM       XC          
 Si      16          
 ---------------------------------------------------------
 Initial plane wave basis and FFT box
 ---------------------------------------------------------
 DONE(2.02587    SEC) : INIT PLANEWAVE
 MEMORY FOR PSI (MB)  : 53.4357
 DONE(2.14139    SEC) : LOCAL POTENTIAL
 DONE(2.16564    SEC) : NON-LOCAL POTENTIAL
 DONE(2.19317    SEC) : INIT BASIS
 -------------------------------------------
 STEP OF RELAXATION : 1
 -------------------------------------------
 START CHARGE      : atomic
 DONE(2.49619    SEC) : INIT SCF
 ITER   ETOT(eV)       EDIFF(eV)      DRHO       TIME(s)    
 CG1    -1.715993e+03  0.000000e+00   4.410e-01  4.290e+00  
 CG2    -1.715532e+03  4.612549e-01   1.895e-01  8.686e-01  
 CG3    -1.715732e+03  -1.998605e-01  4.381e-03  9.360e-01  
 CG4    -1.715743e+03  -1.069036e-02  4.033e-03  1.253e+00  
 CG5    -1.715750e+03  -7.109062e-03  5.397e-05  9.050e-01  
 CG6    -1.715750e+03  -2.715577e-04  5.044e-05  1.349e+00  
 CG7    -1.715750e+03  -1.857458e-04  2.426e-05  8.797e-01  
 CG8    -1.715750e+03  -1.352547e-04  7.944e-05  8.869e-01  
 CG9    -1.715750e+03  6.856684e-04   1.858e-04  9.368e-01  
 CG10   -1.715750e+03  -3.129367e-04  1.255e-05  1.105e+00  
 CG11   -1.715750e+03  1.742551e-06   3.886e-05  1.002e+00  
 CG12   -1.715750e+03  -1.139971e-05  8.168e-06  9.598e-01  
 CG13   -1.715750e+03  -5.350698e-06  9.103e-07  9.003e-01  
 CG14   -1.715750e+03  -3.386112e-06  2.056e-07  1.152e+00  
 CG15   -1.715750e+03  2.312853e-07   1.013e-07  9.660e-01  
 CG16   -1.715750e+03  -6.495043e-08  1.603e-08  9.619e-01  
 CG17   -1.715750e+03  -2.462990e-08  1.429e-09  1.066e+00  
 ><><><><><><><><><><><><><><><><><><><><><><
 TOTAL-STRESS (KBAR):
 ><><><><><><><><><><><><><><><><><><><><><><
 4.127e+01      0.000e+00      7.975e-15      
 0.000e+00      4.127e+01      0.000e+00      
 7.975e-15      0.000e+00      4.127e+01      
 TOTAL-PRESSURE: 4.127e+01 KBAR

  |CLASS_NAME---------|NAME---------------|TIME(Sec)-----|CALLS----|AVG------|PER%-------
                       total               27.893         17        1.6       1e+02     %
   Driver              driver_line         27.842         1         28        1e+02     %
   PW_Basis            setuptransform      0.43315        1         0.43      1.6       %
   ESolver_KS_PW       Run                 20.778         1         21        74        %
   PW_Basis            recip2real          0.94607        99        0.0096    3.4       %
   PW_Basis            gathers_scatterp    0.11274        99        0.0011    0.4       %
   Potential           init_pot            0.26124        1         0.26      0.94      %
   Potential           update_from_charge  4.2893         18        0.24      15        %
   Potential           cal_v_eff           4.2634         18        0.24      15        %
   H_Hartree_pw        v_hartree           0.44038        18        0.024     1.6       %
   PW_Basis            real2recip          1.4507         156       0.0093    5.2       %
   PW_Basis            gatherp_scatters    0.11054        156       0.00071   0.4       %
   PotXC               cal_v_eff           3.7983         18        0.21      14        %
   XC_Functional       v_xc                4.0067         19        0.21      14        %
   Symmetry            rho_symmetry        0.21186        19        0.011     0.76      %
   HSolverPW           solve               15.098         18        0.84      54        %
   Nonlocal            getvnl              2.1001         90        0.023     7.5       %
   pp_cell_vnl         getvnl              1.4305         100       0.014     5.1       %
   WF_igk              get_sk              0.64589        2000      0.00032   2.3       %
   DiagoIterAssist     diagH_subspace      2.3062         85        0.027     8.3       %
   OperatorPW          hPsi                7.3099         13069     0.00056   26        %
   Operator            VeffPW              5.6935         13069     0.00044   20        %
   PW_Basis_K          recip_to_real gpu   3.1628         19435     0.00016   11        %
   PW_Basis_K          real_to_recip gpu   2.966          16554     0.00018   11        %
   Operator            NonlocalPW          1.525          13069     0.00012   5.5       %
   Nonlocal            add_nonlocal_pp     1.2008         13069     9.2e-05   4.3       %
   DiagoIterAssist     LAPACK_subspace     0.79596        85        0.0094    2.9       %
   DiagoCG             diag_once           8.878          90        0.099     32        %
   ElecStatePW         psiToRho            1.1491         18        0.064     4.1       %
   Charge              rho_mpi             0.18609        18        0.01      0.67      %
   Charge              mix_rho             0.57195        16        0.036     2.1       %
   Forces              cal_force_nl        0.62185        1         0.62      2.2       %
   Stress_PW           cal_stress          3.8317         1         3.8       14        %
   Stress_Func         stress_gga          0.12515        1         0.13      0.45      %
   Stress_Func         stres_nl            3.5443         1         3.5       13        %
 ----------------------------------------------------------------------------------------

 START  Time  : Sat Nov 19 22:10:19 2022
 FINISH Time  : Sat Nov 19 22:10:47 2022
 TOTAL  Time  : 28
 SEE INFORMATION IN : OUT.autotest/
[WARNING] yaksa: 1 leaked handle pool objects
[WARNING] yaksa: 1 leaked handle pool objects

Process finished with exit code 0
```