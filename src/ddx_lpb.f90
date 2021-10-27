!> @copyright (c) 2020-2021 RWTH Aachen. All rights reserved.
!!
!! ddX software
!!
!! @file src/dd_core.f90
!! Core routines and parameters of entire ddX software
!!
!! @version 1.0.0
!! @author Abhinav Jha and Michele Nottoli
!! @date 2021-02-25

module ddx_lpb
use ddx_core
use ddx_operators
use ddx_solvers
implicit none
!!
!! Logical variables for iterations,  fcosmo solver, and HSP solver
!!
logical :: first_out_iter
integer :: matAB
!!
!! Hardcoded values
!!
integer :: nbasis0
integer :: lmax0
real(dp),  parameter :: epsp = 1.0d0
!!
!! Taken from Chaoyu's MATLAB code
!!
!! coefvec      : Intermediate value in computation of update_rhs
!! Pchi         : Pchi matrix, Eq. (87)
!! coefY        : Intermediate calculation in Q Matrix Eq. (91)
!! C_ik         : (i'_l0(r_j)/i_l0(r_j)-k'_l0(r_j)/k_l0(r_j))^{-1}
real(dp), allocatable :: coefvec(:,:,:), Pchi(:,:,:), &
                         & coefY(:,:,:)
real(dp), allocatable :: C_ik(:,:)
!! SI_ri        : Bessel' function of first kind
!! DI_ri        : Derivative of Bessel' function of first kind
!! SK_ri        : Bessel' function of second kind
!! DK_ri        : Derivative of Bessel' function of second kind
!! termimat     : i'_l(r_j)/i_l(r_j)
!! tol_gmres    : Tolerance of GMRES iteration
!! n_iter_gmres : Maximum number of GMRES itertation

real(dp), allocatable :: SI_ri(:,:), DI_ri(:,:), SK_ri(:,:), &
                              & DK_ri(:,:), termimat(:,:)
real(dp)              :: tol_gmres, n_iter_gmres

!! Terms related to Forces of ddLPB model
real(dp), allocatable :: diff_re_c1(:,:), diff_re_c2(:,:)
real(dp), allocatable :: diff0_c1(:,:), diff0_c2(:,:)
real(dp), allocatable :: diff_ep_adj(:, :, :), diff_ep_c2(:)
contains
  !!
  !! ddLPB calculation happens here
  !! @param[in] ddx_data : dd Data 
  !! @param[in] phi      : Boundary conditions
  !! @param[in] psi      : Electrostatic potential vector.
  !! @param[in] gradphi  : Gradient of phi
  !! @param[in] hessian  : Hessian of phi
  !! @param[out] esolv   : Electrostatic solvation energy
  !!
  subroutine ddlpb(ddx_data, phi, gradphi, hessian, psi, esolv, force)
  ! main ddLPB
  implicit none
  ! Inputs
  type(ddx_type), intent(inout) :: ddx_data
  real(dp), dimension(ddx_data % ncav), intent(in)       :: phi
  real(dp), dimension(3, ddx_data % ncav), intent(in)    :: gradphi
  real(dp), dimension(3,3, ddx_data % ncav), intent(in)  :: hessian
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: psi
  ! Outputs
  real(dp), intent(out)      :: esolv
  real(dp), dimension(3, ddx_data % nsph), intent(out) :: force
  real(dp), external :: ddot
  logical                    :: converged
  integer                    :: iteration, istatus, ibasis
  real(dp)                   :: inc, old_esolv, sum_gxadj, sum_psix
  !!
  !! Xr         : Reaction potential solution (Laplace equation)
  !! Xe         : Extended potential solution (HSP equation)
  !! rhs_r      : Right hand side corresponding to Laplace equation
  !! rhs_e      : Right hand side corresponding to HSP equation
  !! rhs_r_init : Initial right hand side corresponding to Laplace equation
  !! rhs_e_init : Initial right hand side corresponding to HSP equation
  !! scaled_Xr  : Reaction potential scaled by 1/(4Pi/2l+1)
  !!
  real(dp), allocatable ::   Xr(:,:), Xe(:,:), rhs_r(:,:), rhs_e(:,:), &
                             & rhs_r_init(:,:), rhs_e_init(:,:), &
                             & Xadj_r(:,:), Xadj_e(:,:), &
                             & Xadj_r_sgrid(:,:), Xadj_e_sgrid(:,:), &
                             & scaled_Xr(:,:), diff_re(:,:), normal_hessian_cav(:,:)
  !! g       : Intermediate matrix for computation of g0
  !! f       : Intermediate matrix for computation of f0
  !! g0      : Vector associated to psi_0 Eq.(77) QSM19.SISC
  !! f0      : Vector associated to partial_n_psi_0 Eq.(99) QSM19.SISC
  !! phi_grid: Phi evaluated at grid points
  real(dp), allocatable :: g(:,:), f(:,:), g0(:), f0(:), phi_grid(:, :)
  ! isph    : Index for spheres i
  ! icav    : Index for cavity points
  ! icav_gr : Index for global cavity points for Laplace
  ! icav_ge : Index for global cavity points for HSP
  ! igrid   : Index for grid points
  ! i       : Index for dimension
  ! ok      : Input argument for Jacobi solver
  ! n_iter  : Number of iterative steps
  integer                    :: isph, icav, icav_gr, icav_ge, igrid
  integer                    :: i
  logical                    :: ok
  integer                    :: n_iter
  
  ! Local variables, used in force computation
  ! ef : Electrostatic force using potential of spheres
  real(dp), allocatable :: vsin(:), vcos(:), vplm(:), basloc(:), &
                         & dbasloc(:, :), ef(:, :)
  ! lmax0 set to minimum of 6 or given lmax.
  ! nbasis0 set to minimum of 49 or given (lmax+1)^2.
  ! Previous implementation had hard coded value 6 and 49.
  lmax0 = MIN(6, ddx_data % lmax)
  nbasis0 = MIN(49, ddx_data % nbasis)

  !
  ! Allocate Bessel's functions of the first kind, the second kind, their derivatives,
  ! coefvec, Pchi, and termimat
  !
  call ddlpb_init(ddx_data)

  allocate(Xr(ddx_data % nbasis, ddx_data % nsph),&
           & Xe(ddx_data % nbasis, ddx_data % nsph), &
           & scaled_Xr(ddx_data % nbasis, ddx_data % nsph), &
           & rhs_r(ddx_data % nbasis, ddx_data % nsph), &
           & rhs_e(ddx_data % nbasis, ddx_data % nsph), &
           & rhs_r_init(ddx_data % nbasis, ddx_data % nsph),&
           & rhs_e_init(ddx_data % nbasis, ddx_data % nsph), &
           & Xadj_r(ddx_data % nbasis, ddx_data % nsph),&
           & Xadj_e(ddx_data % nbasis, ddx_data % nsph), &
           & Xadj_r_sgrid(ddx_data % ngrid, ddx_data % nsph), &
           & Xadj_e_sgrid(ddx_data % ngrid, ddx_data % nsph), &
           & diff_re(ddx_data % nbasis, ddx_data % nsph), &
           & normal_hessian_cav(3,ddx_data % ncav), &
           & g(ddx_data % ngrid,ddx_data % nsph),&
           & f(ddx_data % ngrid, ddx_data % nsph), &
           & g0(ddx_data % nbasis),&
           & f0(ddx_data % nbasis),&
           & phi_grid(ddx_data % ngrid, ddx_data % nsph), &
           & vsin(ddx_data % lmax+1), vcos(ddx_data % lmax+1), &
           & vplm(ddx_data % nbasis), basloc(ddx_data % nbasis), &
           & dbasloc(3, ddx_data % nbasis), &
           & ef(3, ddx_data % nsph), &
           & stat = istatus)
  if (istatus.ne.0) write(6,*) 'ddlpb allocation failed'

  !! Setting initial values to zero
  Xr = zero ; Xe = zero; scaled_Xr = zero
  rhs_r = zero ; rhs_e = zero
  rhs_r_init = zero ; rhs_e_init = zero
  Xadj_r = zero ; Xadj_e = zero
  Xadj_r_sgrid = zero; Xadj_e_sgrid = zero
  diff_re = zero; normal_hessian_cav = zero
  g = zero; f = zero; g0 = zero; f0 = zero
  phi_grid = zero; vsin = zero; vcos = zero; vplm = zero
  basloc = zero; dbasloc = zero; ef = zero
  inc = zero; old_esolv = zero
  force = zero

  !! wghpot: Weigh potential at cavity points. Comes from ddCOSMO
  !!         Intermediate computation of G_0 Eq.(77) QSM19.SISC
  !!
  !! @param[in]  phi      : Boundary conditions (This is psi_0 Eq.(20) QSM19.SISC)
  !! @param[out] phi_grid : Phi evaluated at grid points
  !! @param[out] g        : Boundary conditions on solute-solvent boundary gamma_j_e
  !!
  call wghpot(ddx_data, phi, phi_grid, g)
  !!
  !! wghpot_f : Intermediate computation of F_0 Eq.(75) from QSM19.SISC
  !!
  !! @param[in]  gradphi : Gradient of psi_0
  !! @param[out] f       : Boundary conditions scaled by characteristic function
  !!
  call wghpot_f(ddx_data, gradphi,f)

  !!
  !! Integrate Right hand side
  !! rhs_r_init: g0+f0
  !! rhs_e_init: f0
  !!
  do isph = 1, ddx_data % nsph
    !! intrhs is a subroutine in ddx_operators
    !! @param[in]  isph : Sphere number, used for output
    !! @param[in]  g    : Intermediate right side g
    !! @param[out] g0   : Integrated right side Eq.(77) in QSM19.SISC
    call intrhs(ddx_data % iprint, ddx_data % ngrid, &
                ddx_data % lmax, ddx_data % vwgrid, ddx_data % vgrid_nbasis, &
                & isph, g(:,isph), g0)
    call intrhs(ddx_data % iprint, ddx_data % ngrid, &
                ddx_data % lmax, ddx_data % vwgrid, ddx_data % vgrid_nbasis, &
                & isph,f(:,isph),f0)
    !! rhs 
    rhs_r_init(:,isph) = g0 + f0
    rhs_e_init(:,isph) = f0
  end do

  rhs_r = rhs_r_init
  rhs_e = rhs_e_init
  
  n_iter = ddx_data % maxiter
  
  ! Set values to default values
  first_out_iter = .true.
  converged = .false.
  ok = .false.
  iteration = one

  do while (.not.converged)

    !! Solve the ddCOSMO step
    !! A X_r = RHS_r (= G_X+G_0) 
    !! Call Jacobi solver
    !! @param[in]      nsph*nylm : Size of matrix
    !! @param[in]      iprint    : Flag for printing
    !! @param[in]      ndiis     : Number of points to be used for 
    !!                            DIIS extrapolation. Set to 25 in ddCOSMO
    !! @param[in]      4         : Norm to be used to evaluate convergence
    !!                             4 refers to user defined norm. Here hnorm
    !! @param[in]      tol       : Convergence tolerance
    !! @param[in]      rhs_r     : Right-hand side
    !! @param[in, out] xr        : Initial guess to solution and final solution
    !! @param[in, out] n_iter    : Number of iterative steps
    !! @param[in, out] ok        : Boolean to check whether the solver converged
    !! @param[in]      lx        : External subroutine to compute matrix 
    !!                             multiplication, i.e., Lx_r, comes from matvec.f90
    !! @param[in]      ldm1x     : External subroutine to apply invert diagonal
    !!                             matrix to vector, i.e., L^{-1}x_r, comes from matvec.f90
    !! @param[in]      hnorm     : User defined norm, comes from matvec.f90
    call jacobi_diis(ddx_data, ddx_data % n, ddx_data % iprint, ddx_data % ndiis, &
                     & 4, ddx_data % tol, rhs_r, Xr, n_iter, ok, lx, ldm1x, hnorm)
    ! Scale by the factor of (2l+1)/4Pi
    call convert_ddcosmo(ddx_data, 1, Xr)
    !converged = .true.

    !! Solve ddLPB step
    !! B X_e = RHS_e (= F_0)
    call lpb_hsp(ddx_data, rhs_e, Xe)

    !! Update the RHS
    !! / RHS_r \ = / g + f \ - / c1 c2 \ / X_r \
    !! \ RHS_e /   \ f     /   \ c1 c2 / \ X_e /
    call update_rhs(ddx_data, rhs_r_init, rhs_e_init, rhs_r, rhs_e, Xr, Xe)
    ! call print_ddvector('rhs_r',rhs_r)
    ! call print_ddvector('rhs_e',rhs_e)

    !! Compute energy
    !! esolv = pt5*sprod(nsph*nylm,xr,psi)
    esolv = zero
    do isph = 1, ddx_data % nsph
      esolv = esolv + pt5*ddx_data % charge(isph)*Xr(1,isph)*(one/sqrt4pi)
    end do

    !! Check for convergence
    inc = zero
    inc = abs(esolv - old_esolv)/abs(esolv)
    old_esolv = esolv
    if ((iteration.gt.1) .and. (inc.lt.ddx_data % tol)) then
      write(6,*) 'Reach tolerance.'
      converged = .true.
    end if
    write(6,*) iteration, esolv, inc
    iteration = iteration + 1

    ! to be removed
    first_out_iter = .false.
  end do

  ! Start the Force computation
  if(ddx_data % force .eq. 1) then
    write(*,*) 'Computation of Forces for ddLPB'
    ! Call the subroutine adjoint to solve the adjoint solution
    call ddx_lpb_adjoint(ddx_data, psi, Xadj_r, Xadj_e)
    ! Debug check the inner product
    sum_gxadj = zero
    sum_psix = zero

    do ibasis = 1, ddx_data % nbasis
      do isph = 1, ddx_data % nsph
        sum_gxadj = sum_gxadj + &
                              & Xadj_r(ibasis, isph)*rhs_r_init(ibasis, isph) + &
                              & Xadj_e(ibasis, isph)*rhs_e_init(ibasis, isph)
        sum_psix = sum_psix + &
                            & Xr(ibasis, isph)*psi(ibasis, isph)
      end do
    end do
    write(*,*) '<g,Xadj> : ', sum_gxadj, ' , <X,Psi> : ', sum_psix

    ! Compute the derivative of the normal derivative of psi_0
    icav = 0
    do isph = 1, ddx_data % nsph
      do igrid = 1, ddx_data % ngrid
        if(ddx_data % ui(igrid, isph) .gt. zero) then
          icav = icav + 1
          do i = 1, 3
            normal_hessian_cav(:, icav) = normal_hessian_cav(:,icav) +&
                                    & hessian(:,i,icav)*ddx_data % cgrid(i,igrid)
          end do
        end if
      end do
    end do

    ! Call dgemm to integrate the adjoint solution on the grid points
    call dgemm('T', 'N', ddx_data % ngrid, ddx_data % nsph, &
            & ddx_data % nbasis, one, ddx_data % vgrid, ddx_data % vgrid_nbasis, &
            & Xadj_r , ddx_data % nbasis, zero, Xadj_r_sgrid, &
            & ddx_data % ngrid)
    call dgemm('T', 'N', ddx_data % ngrid, ddx_data % nsph, &
            & ddx_data % nbasis, one, ddx_data % vgrid, ddx_data % vgrid_nbasis, &
            & Xadj_e , ddx_data % nbasis, zero, Xadj_e_sgrid, &
            & ddx_data % ngrid)

    ! Scale by the factor of 1/(4Pi/(2l+1))
    scaled_Xr = Xr
    call convert_ddcosmo(ddx_data, -1, scaled_Xr)

    do isph = 1, ddx_data % nsph
      ! Compute A^k*Xadj_r, using Subroutine from ddCOSMO
      call fdoka(ddx_data, isph, scaled_Xr, Xadj_r_sgrid(:, isph), &
                & basloc, dbasloc, vplm, vcos, vsin, force(:,isph))
      call fdokb(ddx_data, isph, scaled_Xr, Xadj_r_sgrid, basloc, &
                & dbasloc, vplm, vcos, vsin, force(:, isph))
      ! Compute B^k*Xadj_e
      call fdoka_b_xe(ddx_data, isph, Xe, Xadj_e_sgrid(:, isph), &
                & basloc, dbasloc, vplm, vcos, vsin, force(:,isph))
      call fdokb_b_xe(ddx_data, isph, Xe, Xadj_e_sgrid, &
                & basloc, dbasloc, vplm, vcos, vsin, force(:, isph))
      ! Compute C1 and C2 contributions
      diff_re = zero
      call fdouky(ddx_data, isph, &
                  & Xr, Xe, &
                  & Xadj_r_sgrid, Xadj_e_sgrid, &
                  & Xadj_r, Xadj_e, &
                  & force(:, isph), &
                  & diff_re)
      call derivative_P(ddx_data, isph,&
                  & Xr, Xe, &
                  & Xadj_r_sgrid, Xadj_e_sgrid, &
                  & diff_re, &
                  & force(:, isph))
      ! Computation of G0
      call fdoga(ddx_data, isph, Xadj_r_sgrid, phi_grid, force(:, isph))
    end do
    ! Computation of G0 continued
    ! NOTE: fdoga returns a positive summation
    force = -force
    icav = 0
    do isph = 1, ddx_data % nsph
      do igrid = 1, ddx_data % ngrid
        if(ddx_data % ui(igrid, isph) .ne. zero) then
          icav = icav + 1
          ddx_data % zeta(icav) = -ddx_data % wgrid(igrid) * &
                        & ddx_data % ui(igrid, isph) * ddot(ddx_data % nbasis, &
                        & ddx_data % vgrid(1, igrid), 1, &
                        & Xadj_r(1, isph), 1)
          force(:, isph) = force(:, isph) + &
                        & ddx_data % zeta(icav)*gradphi(:, icav)
        end if
      end do
    end do
    call efld(ddx_data % ncav, ddx_data % zeta, ddx_data % ccav, &
                & ddx_data % nsph, ddx_data % csph, ef)
    do isph = 1, ddx_data % nsph
      force(:, isph) = force(:, isph) - ef(:, isph)*ddx_data % charge(isph)
    end do

    icav_gr = zero
    icav_ge = zero
    do isph = 1, ddx_data % nsph
      ! Computation of F0
      call fdouky_f0(ddx_data, isph, Xadj_r, Xadj_r_sgrid, &
          & gradphi, force(:, isph))
      call fdoco(ddx_data, isph, Xadj_r_sgrid, gradphi, &
             & normal_hessian_cav, icav_gr, force(:, isph))
      ! Computation of F0
      call fdouky_f0(ddx_data, isph, Xadj_e, Xadj_e_sgrid, &
          & gradphi, force(:, isph))
      call fdoco(ddx_data, isph, Xadj_e_sgrid, gradphi, &
             & normal_hessian_cav, icav_ge, force(:, isph))
    end do

    force = pt5*force
  endif
  call ddlpb_free(ddx_data)
  deallocate(Xr, Xe, scaled_Xr, &
           & rhs_r, rhs_e, &
           & rhs_r_init, rhs_e_init, &
           & Xadj_r, Xadj_e, &
           & Xadj_r_sgrid, Xadj_e_sgrid, diff_re, &
           & normal_hessian_cav, g, f, g0, f0, phi_grid, &
           & vsin, vcos, vplm, basloc, dbasloc, ef, stat = istatus)
  if (istatus.ne.0) write(6,*) 'ddlpb deallocation failed'


  return
  end subroutine ddlpb
  !!
  !! Allocate Bessel's functions of the first kind and the second kind
  !! Uses the file bessel.f90
  !! @param[out] SI_ri : Bessel's function of the first kind
  !! @param[out] DI_ri : Derivative of Bessel's function of the first kind
  !! @param[out] SK_ri : Bessel's function of the second kind
  !! @param[out] DK_ri : Derivative of Bessel's function of the second kind
  !!
  subroutine ddlpb_init(ddx_data)
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  integer                     :: istatus, isph, igrid, ind, icav, l, l0, m0, ind0, jsph
  real(dp)                    :: termi, termk, term, rho, ctheta, stheta, cphi, sphi, rijn
  real(dp), dimension(3)      :: sijn ,vij
  real(dp), dimension(ddx_data % nbasis) :: basloc, vplm
  real(dp), dimension(ddx_data % lmax+1) :: vcos, vsin
  real(dp), dimension(0:lmax0) :: SK_rijn, DK_rijn
  allocate(SI_ri(0:ddx_data % lmax, ddx_data % nsph),&
           & DI_ri(0:ddx_data % lmax, ddx_data % nsph),&
           & SK_ri(0:ddx_data % lmax, ddx_data % nsph), &
           & DK_ri(0:ddx_data % lmax, ddx_data % nsph), &
           & diff_re_c1(ddx_data % nbasis, ddx_data % nsph), &
           & diff_re_c2(ddx_data % nbasis, ddx_data % nsph), &
           & diff0_c1(nbasis0, ddx_data % nsph), &
           & diff0_c2(nbasis0, ddx_data % nsph), &
           & diff_ep_adj(ddx_data % ncav, ddx_data % nbasis, ddx_data % nsph), &
           & diff_ep_c2(ddx_data % ncav), &
           & coefvec(ddx_data % ngrid, ddx_data % nbasis, ddx_data % nsph), &
           & Pchi(ddx_data % nbasis, nbasis0, ddx_data % nsph), &
           & coefY(ddx_data % ncav, nbasis0, ddx_data % nsph), &
           & C_ik(0:ddx_data % lmax, ddx_data % nsph), &
           & termimat(0:ddx_data % lmax, ddx_data % nsph), stat=istatus)

  if (istatus.ne.0) then
    write(*,*)'ddlpb_init : [1] allocation failed !'
    stop
  end if
  ! Set for GMRES
  matAB = 1
  tol_gmres = 1e-8

  SK_rijn = zero
  DK_rijn = zero
  do isph = 1, ddx_data % nsph
    call modified_spherical_bessel_first_kind(ddx_data % lmax, &
                     & ddx_data % rsph(isph)*ddx_data % kappa,&
                     & SI_ri(:,isph), &
                     & DI_ri(:,isph))
    call modified_spherical_bessel_second_kind(ddx_data % lmax, &
                     & ddx_data % rsph(isph)*ddx_data % kappa, &
                     & SK_ri(:,isph), &
                     & DK_ri(:,isph))
    ! Compute matrix PU_i^e(x_in)
    ! Previous implementation in update_rhs. Made it in ddinit, so as to use
    ! it in Forces as well.
    call mkpmat(ddx_data, isph, Pchi(:,:,isph))
    ! Compute w_n*Ui(x_in)*Y_lm(s_n)
    do igrid = 1,ddx_data % ngrid
      if (ddx_data % ui(igrid, isph) .gt. 0) then
        do ind  = 1, ddx_data % nbasis
          coefvec(igrid,ind,isph) = ddx_data % wgrid(igrid)*&
                                  & ddx_data % ui(igrid,isph)*&
                                  & ddx_data % vgrid(ind,igrid)
        end do
      end if
    end do
    ! Compute i'_l(r_i)/i_l(r_i)
    do l = 0, ddx_data % lmax
      termimat(l,isph) = DI_ri(l,isph)/SI_ri(l,isph)*ddx_data % kappa
    end do
    ! Compute (i'_l0/i_l0 - k'_l0/k_l0)^(-1) is computed in Eq.(97)
    do l0 = 0, lmax0
      termi = DI_ri(l0,isph)/SI_ri(l0,isph)*ddx_data % kappa
      termk = DK_ri(l0,isph)/SK_ri(l0,isph)*ddx_data % kappa
      C_ik(l0, isph) = one/(termi - termk)
    end do
  end do

  icav = zero
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if(ddx_data % ui(igrid, isph) .gt. zero) then
        icav = icav + 1
        do jsph = 1, ddx_data % nsph
          vij  = ddx_data % csph(:,isph) + &
               & ddx_data % rsph(isph)*ddx_data % cgrid(:,igrid) - &
               & ddx_data % csph(:,jsph)
          rijn = sqrt(dot_product(vij,vij))
          sijn = vij/rijn

          ! Compute Bessel function of 2nd kind for the coordinates
          ! (s_ijn, r_ijn) and compute the basis function for s_ijn
          call modified_spherical_bessel_second_kind(lmax0, rijn*ddx_data % kappa,&
                                                   & SK_rijn, DK_rijn)
          !call SPHK_bessel(lmax0,rijn*ddx_data % kappa,NM,SK_rijn,DK_rijn)
          call ylmbas(sijn , rho, ctheta, stheta, cphi, &
                      & sphi, ddx_data % lmax, ddx_data % vscales, &
                      & basloc, vplm, vcos, vsin)
          do l0 = 0, lmax0
            term = SK_rijn(l0)/SK_ri(l0,jsph)
            do m0 = -l0,l0
              ind0 = l0*l0 + l0 + m0 + 1
              coefY(icav, ind0, jsph) = C_ik(l0,jsph)*term*basloc(ind0)
            end do
          end do
        end do
      end if
    end do
  end do
  return
  end subroutine ddlpb_init


  !!
  !! Find intermediate F0 in the RHS of the ddLPB model given in Eq.(82)
  !! @param[in]  gradphi : Gradient of psi_0
  !! @param[out] f       : Intermediate calculation of F0
  !!
  subroutine wghpot_f(ddx_data, gradphi, f )
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  real(dp), dimension(3, ddx_data % ncav),       intent(in)  :: gradphi
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph),    intent(out) :: f

  integer :: isph, ig, ic, ind, ind0, jg, l, m, jsph
  real(dp) :: nderphi, sumSijn, rijn, coef_Ylm, sumSijn_pre, termi, &
      & termk, term
  real(dp), dimension(3) :: sijn, vij
  real(dp) :: rho, ctheta, stheta, cphi, sphi
  real(dp), allocatable :: SK_rijn(:), DK_rijn(:)

  integer :: l0, m0, NM, kep, istatus
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph) :: c0
  real(dp), allocatable :: vplm(:), basloc(:), vcos(:), vsin(:)

  ! initialize
  allocate(vplm(ddx_data % nbasis),basloc(ddx_data % nbasis),vcos(ddx_data % lmax+1),vsin(ddx_data % lmax+1))
  allocate(SK_rijn(0:lmax0),DK_rijn(0:lmax0))
  ic = 0 ; f(:,:)=0.d0
  !
  ! Compute c0 Eq.(98) QSM19.SISC
  !
  do isph = 1, ddx_data % nsph
    do ig = 1, ddx_data % ngrid
      if ( ddx_data % ui(ig,isph).ne.zero ) then
        ic = ic + 1
        nderphi = dot_product( gradphi(:,ic),ddx_data % cgrid(:,ig) )
        c0(:, isph) = c0(:,isph) + &
                     & ddx_data % wgrid(ig)*ddx_data % ui(ig,isph)*&
                     & nderphi*ddx_data % vgrid(:,ig)
      end if
    end do
  end do


  ! Computation of F0 using above terms
  ! kep: External grid poitns
  kep = 0
  do isph = 1, ddx_data % nsph
    do ig = 1, ddx_data % ngrid
      if (ddx_data % ui(ig,isph).gt.zero) then
        kep = kep + 1
        sumSijn = zero
        ! Loop to compute Sijn
        do jsph = 1, ddx_data % nsph
          sumSijn_pre = sumSijn
          vij  = ddx_data % csph(:,isph) + ddx_data % rsph(isph)*ddx_data % cgrid(:,ig) - ddx_data % csph(:,jsph)
          rijn = sqrt(dot_product(vij,vij))
          sijn = vij/rijn
          
          ! Compute Bessel function of 2nd kind for the coordinates
          ! (s_ijn, r_ijn) and compute the basis function for s_ijn
          call modified_spherical_bessel_second_kind(lmax0, rijn*ddx_data % kappa, SK_rijn, DK_rijn)
          !call SPHK_bessel(lmax0,rijn*ddx_data % kappa,NM,SK_rijn,DK_rijn)
          call ylmbas(sijn , rho, ctheta, stheta, cphi, &
                      & sphi, ddx_data % lmax, ddx_data % vscales, &
                      & basloc, vplm, vcos, vsin)

          do l0 = 0,lmax0
            term = SK_rijn(l0)/SK_ri(l0,jsph)
            ! coef_Ylm : (der_i_l0/i_l0 - der_k_l0/k_l0)^(-1)*k_l0(r_ijn)/k_l0(r_i)
            coef_Ylm =  C_ik(l0,jsph)*term
            do m0 = -l0, l0
              ind0 = l0**2 + l0 + m0 + 1
              sumSijn = sumSijn + c0(ind0,jsph)*coef_Ylm*basloc(ind0)
              !coefY(kep,ind0,jsph) = coef_Ylm*basloc(ind0)
            end do
          end do
        end do
        !
        ! Here Intermediate value of F_0 is computed Eq. (99)
        ! Mutilplication with Y_lm and weights will happen afterwards
        !write(6,*) sumSijn, epsp, eps, ddx_data % ui(ig,isph)
        f(ig,isph) = -(epsp/ddx_data % eps)*ddx_data % ui(ig,isph) * sumSijn
      end if
    end do
  end do 

  deallocate( vplm, basloc, vcos, vsin, SK_rijn, DK_rijn  )
  return
  end subroutine wghpot_f

  !
  ! Subroutine used for the GMRES solver
  ! NOTE: It is refered as matABx in the GMRES solver.
  !       Fortran is not case sensitive
  ! @param[in]      n : Size of the matrix
  ! @param[in]      x : Input vector
  ! @param[in, out] y : y=A*x
  !
  subroutine matABx(ddx_data, n, x, y )
  implicit none 
  type(ddx_type), intent(in)  :: ddx_data
  integer, intent(in) :: n
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: x
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(inout) :: y
  integer :: isph, istatus
  real(dp), allocatable :: pot(:), vplm(:), basloc(:), vcos(:), vsin(:)
  integer :: i
  ! allocate workspaces
  allocate( pot(ddx_data % ngrid), vplm(ddx_data % nbasis), basloc(ddx_data % nbasis), &
            & vcos(ddx_data % lmax+1), vsin(ddx_data % lmax+1), stat=istatus )
  if ( istatus.ne.0 ) then
    write(*,*) 'Bx: allocation failed !'
    stop
  endif
  
  if (ddx_data % iprint .ge. 5) then
      call prtsph('X', ddx_data % nbasis, ddx_data % lmax, ddx_data % nsph, 0, &
          & x)
  end if

  y = zero
  do isph = 1, ddx_data % nsph
    call calcv2_lpb(ddx_data, isph, pot, x, basloc, vplm, vcos, vsin )
    ! intrhs comes from ddCOSMO
    call intrhs(ddx_data % iprint, ddx_data % ngrid, &
                ddx_data % lmax, ddx_data % vwgrid, ddx_data % vgrid_nbasis, &
                & isph, pot, y(:,isph) )
    ! Action of off-diagonal blocks
    y(:,isph) = - y(:,isph)
    ! Add action of diagonal block
    y(:,isph) = y(:,isph) + x(:,isph)
  end do
  
  if (ddx_data % iprint .ge. 5) then
      call prtsph('Bx (off-diagonal)', ddx_data % nbasis, ddx_data % lmax, &
          & ddx_data % nsph, 0, y)
  end if
  deallocate( pot, basloc, vplm, vcos, vsin , stat=istatus )
  if ( istatus.ne.0 ) then
    write(*,*) 'matABx: allocation failed !'
    stop
  endif
  end subroutine matABx

  !!
  !! Scale the ddCOSMO solution vector
  !! @param[in]      direction : Direction of the scaling
  !! @param[in, out] vector    : ddCOSMO solution vector
  !!
  subroutine convert_ddcosmo(ddx_data, direction, vector)
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  integer, intent(in) :: direction
  real(dp), intent(inout) :: vector(ddx_data % nbasis, ddx_data % nsph)
  integer :: isph, l, m, ind
  real(dp) :: fac
  
  do isph = 1, ddx_data % nsph
    do l = 0, ddx_data % lmax
      ind = l*l + l + 1
      fac = four*pi/(two*dble(l) + one) 
      if (direction.eq.-1) fac = one/fac
      do m = -l, l
        vector(ind + m,isph) = fac*vector(ind + m,isph)
      end do
    end do
  end do
  return
  end subroutine convert_ddcosmo

  !!
  !! Solve the HSP problem
  !! @param[in]      rhs : Right hand side for HSP
  !! @param[in, out] Xe  : Solution vector
  !!
  subroutine lpb_hsp(ddx_data, rhs, Xe)
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: rhs
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(inout) :: Xe
  integer :: isph, info
  real(dp) :: r_norm
  integer, parameter  :: gmm = 20, gmj = 25
  real(dp), dimension(ddx_data % nsph*ddx_data % nbasis, 0:2*gmj+gmm+2-1) :: work
  
  work = zero
  Xe = rhs
  
  !!
  !! Call GMRES solver
  !! @param[in]      Residue_print : Set to false by default. Prints the
  !!                                 intermediate residue.
  !! @param[in]      nsph*nylm     : Size of matrix
  !! @param[in]      gmj           : Integer truncation parameter, Default: 25
  !! @param[in]      gmm           : Integer dimension of the GMRES,
  !!                                 Default: 20
  !! @param[in]      rhs           : Right hand side
  !! @param[in, out] Xe            : Initial guess of the problem
  !! @param[in]      work          : Work space of size
  !!                               : nsph*nylm X (2*gmj + gmm + 2)
  !! @param[in]      tol_gmres     : GMRES tolerance
  !! @param[in]      Stopping      : Stopping criteria, Default set to
  !!                                 'rel' for relative. Other option
  !!                                 'abs' for absolute
  !! @param[in]      n_iter_gmres  : Number of GMRES iteration
  !! @param[in]      r_norm        : Residual measure
  !! @param[in]      matABx        : Subroutine A*x. Named matabx in file
  !! @param[in, out] info          : Flag after solve. 0 means within tolerance
  !!                                 1 means max number of iteration
  call gmresr(ddx_data, .false., ddx_data % nsph*ddx_data % nbasis, gmj, gmm, & 
             & rhs, Xe, work, tol_gmres,'rel', n_iter_gmres, r_norm, matABx, info)

  endsubroutine lpb_hsp

  !
  ! Intermediate computation of BX_e
  ! @param[in]      isph   : Number of the sphere
  ! @param[in, out] pot    : Array of size ngrid
  ! @param[in]      x      : Input vector (Usually X_e)
  ! @param[in, out] basloc : Used to compute spherical harmonic
  ! @param[in, out] vplm   : Used to compute spherical harmonic
  ! @param[in, out] vcos   : Used to compute spherical harmonic
  ! @param[in, out] vsin   : Used to compute spherical harmonic
  !
  subroutine calcv2_lpb (ddx_data, isph, pot, x, basloc, vplm, vcos, vsin )
  type(ddx_type), intent(in) :: ddx_data
  integer, intent(in) :: isph
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: x
  real(dp), dimension(ddx_data % ngrid), intent(inout) :: pot
  real(dp), dimension(ddx_data % nbasis), intent(inout) :: basloc
  real(dp), dimension(ddx_data % nbasis), intent(inout) :: vplm
  real(dp), dimension(ddx_data % lmax+1), intent(inout) :: vcos
  real(dp), dimension(ddx_data % lmax+1), intent(inout) :: vsin
  real(dp), dimension(ddx_data % nbasis) :: fac_cosmo, fac_hsp
  integer :: its, ij, jsph
  real(dp) :: rho, ctheta, stheta, cphi, sphi
  real(dp) :: vij(3), sij(3)
  real(dp) :: vvij, tij, xij, oij

  pot = zero
  do its = 1, ddx_data % ngrid
    if (ddx_data % ui(its,isph).lt.one) then
      do ij = ddx_data % inl(isph), ddx_data % inl(isph+1)-1
        jsph = ddx_data % nl(ij)

        ! compute geometrical variables
        vij  = ddx_data % csph(:,isph) + ddx_data % rsph(isph)*ddx_data % cgrid(:,its) - ddx_data % csph(:,jsph)
        vvij = sqrt(dot_product(vij,vij))
        tij  = vvij/ddx_data % rsph(jsph)
        if ( tij.lt.one ) then
          sij = vij/vvij
          call ylmbas(sij, rho, ctheta, stheta, cphi, &
                      & sphi, ddx_data % lmax, &
                      & ddx_data % vscales, basloc, &
                      & vplm, vcos, vsin)
          call inthsp(ddx_data, vvij, ddx_data % rsph(jsph), jsph, basloc, fac_hsp)
          xij = fsw(tij, ddx_data % se, ddx_data % eta)
          if (ddx_data % fi(its,isph).gt.one) then
            oij = xij/ddx_data % fi(its, isph)
          else
            oij = xij
          end if
          pot(its) = pot(its) + oij*dot_product(fac_hsp,x(:,jsph))
        end if
      end do
    end if
  end do
  endsubroutine calcv2_lpb


  !
  ! Intermediate calculation in calcv2_lpb subroutine
  ! @param[in]  rijn    : Radius of sphers x_ijn
  ! @param[in]  ri      : Radius of sphers x_i
  ! @param[in]  isph    : Index of sphere
  ! @param[in]  basloc  : Spherical Harmonic
  ! @param[out] fac_hsp : Return bessel function ratio multiplied by 
  !                       the spherical harmonic Y_l'm'. Array of size nylm
  !
  subroutine inthsp(ddx_data, rijn, ri, isph, basloc, fac_hsp)
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  integer, intent(in) :: isph
  real(dp), intent(in) :: rijn, ri
  real(dp), dimension(ddx_data % nbasis), intent(in) :: basloc
  real(dp), dimension(ddx_data % nbasis), intent(inout) :: fac_hsp
  real(dp), dimension(0:ddx_data % lmax) :: SI_rijn, DI_rijn
  integer :: l, m, ind, NM

  SI_rijn = 0
  DI_rijn = 0
  fac_hsp = 0

  ! Computation of modified spherical Bessel function values      
  !call SPHI_bessel(ddx_data % lmax,rijn*ddx_data % kappa,NM,SI_rijn,DI_rijn)
  call modified_spherical_bessel_first_kind(ddx_data % lmax, rijn*ddx_data % kappa, SI_rijn, DI_rijn)
  
  do l = 0, ddx_data % lmax
    do  m = -l, l
      ind = l*l + l + 1 + m
      fac_hsp(ind) = SI_rijn(l)/SI_ri(l,isph)*basloc(ind)
    end do
  end do
  endsubroutine inthsp


  subroutine intcosmo(ddx_data, tij, basloc, fac_cosmo)
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  real(dp),  intent(in) :: tij
  real(dp), dimension(ddx_data % nbasis), intent(in) :: basloc
  real(dp), dimension(ddx_data % nbasis), intent(inout) :: fac_cosmo
  integer :: l, m, ind
  do l = 0, ddx_data % lmax
    do  m = -l, l
        ind = l*l + l + 1 + m
        fac_cosmo(ind) = tij**l*basloc(ind)
    end do
  end do
  end subroutine intcosmo

  !
  ! Update the RHS in outer iteration
  ! @param[in] rhs_cosmo_init : G_0
  ! @param[in] rhs_hsp_init   : F_0
  ! @param[in, out] rhs_cosmo : -C_1*X_r^(k-1) - C_2*X_e^(k-1) + G_0 + F_0
  ! @param[in, out] rhs_hsp   : -C_1*X_r^(k-1) - C_2*X_e^(k-1) + F_0
  ! @param[in] Xr             : X_r^(k-1)
  ! @param[in] Xe             : X_e^(k-1)
  !
  subroutine update_rhs(ddx_data, rhs_cosmo_init, rhs_hsp_init, rhs_cosmo, & 
      & rhs_hsp, Xr, Xe)
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(in) :: rhs_cosmo_init, &
      & rhs_hsp_init
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(inout) :: rhs_cosmo, rhs_hsp
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph) :: rhs_plus
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(in) :: Xr, Xe
  integer :: isph, jsph, igrid, kep, ind, l,m, ind0, istatus
  real(dp), dimension(3) :: vij
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph) :: diff_re_c1_c2
  real(dp), dimension(nbasis0,ddx_data % nsph) :: diff0
  real(dp), dimension(ddx_data % nbasis,ddx_data % nbasis,ddx_data % nsph) :: smat
  real(dp), dimension(ddx_data % ncav) :: diff_ep
  real(dp) :: Qval, rijn, val
  integer :: c0, cr, c_qmat, c_init, c_ep0, c_ep1 !, nbasis_appro
      
  ! diff_re = epsp/eps*l1/ri*Xr - i'(ri)/i(ri)*Xe,
  diff_re_c1_c2 = zero
  do jsph = 1, ddx_data % nsph
    do l = 0, ddx_data % lmax
      do m = -l,l
        ind = l**2 + l + m + 1
        diff_re_c1_c2(ind,jsph) = ((epsp/ddx_data % eps)*&
                                & (l/ddx_data % rsph(jsph)) * &
                                & Xr(ind,jsph) - termimat(l,jsph)*Xe(ind,jsph))
      end do
    end do
  end do

  ! diff0 = Pchi * diff_er, linear scaling
  ! TODO: probably doing PX on the fly is better 
  diff0 = zero 
  do jsph = 1, ddx_data % nsph
    do ind0 = 1, nbasis0
      diff0(ind0, jsph) = dot_product(diff_re_c1_c2(:,jsph), &
          & Pchi(:,ind0, jsph))
    end do
  end do

  ! diff_ep = diff0 * coefY,    COST: M^2*nbasis*Nleb
  diff_ep = zero
  do kep = 1, ddx_data % ncav
    val = zero
    do jsph = 1, ddx_data % nsph 
      do ind0 = 1, nbasis0
        val = val + diff0(ind0,jsph)*coefY(kep,ind0,jsph)
      end do
    end do
    diff_ep(kep) = val 
  end do

  rhs_plus = zero
  kep = 0
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if (ddx_data % ui(igrid,isph).gt.zero) then
        kep = kep + 1
        do ind = 1, ddx_data % nbasis
          rhs_plus(ind,isph) = rhs_plus(ind,isph) + &
              & coefvec(igrid,ind,isph)*diff_ep(kep)
        end do
      end if
    end do
  end do

  rhs_cosmo = rhs_cosmo_init - rhs_plus
  rhs_hsp = rhs_hsp_init - rhs_plus

  return
  end subroutine update_rhs  

  !
  ! Computation of P_chi
  ! @param[in]  isph : Sphere number
  ! @param[out] pmat : Matrix of size nbasis X (lmax0+1)^2, Fixed lmax0
  !
  subroutine mkpmat(ddx_data, isph, pmat)
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  integer,  intent(in) :: isph
  real(dp), dimension(ddx_data % nbasis, (lmax0+1)**2), intent(inout) :: pmat
  integer :: l, m, ind, l0, m0, ind0, its, nbasis0
  real(dp)  :: f, f0
  pmat(:,:) = zero
  do its = 1, ddx_data % ngrid
    if (ddx_data % ui(its,isph).ne.0) then
      do l = 0, ddx_data % lmax
        ind = l*l + l + 1
        do m = -l,l
          f = ddx_data % wgrid(its) * ddx_data % vgrid(ind+m,its) &
             & * ddx_data % ui(its,isph)
          do l0 = 0, lmax0
            ind0 = l0*l0 + l0 + 1
            do m0 = -l0, l0
              f0 = ddx_data % vgrid(ind0+m0,its)
              pmat(ind+m,ind0+m0) = pmat(ind+m,ind0+m0) + f * f0
            end do
          end do
        end do
      end do
    end if
  end do
  endsubroutine mkpmat

  !
  ! Computation of Adjoint
  ! @param[in] ddx_data: Input data file
  ! @param[in] psi     : psi_r
  subroutine ddx_lpb_adjoint(ddx_data, psi, Xadj_r, Xadj_e)
  implicit none
  type(ddx_type), intent(in)           :: ddx_data
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: psi
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(out) :: Xadj_r, Xadj_e
  real(dp), external :: dnrm2
  ! Local Variables
  ! ok             : Logical expression used in Jacobi solver
  ! istatus        : Status of allocation
  ! isph           : Index for the sphere
  ! ibasis         : Index for the basis
  ! iteration      : Number of outer loop iterations
  ! inc            : Check for convergence threshold
  ! relative_num   : Numerator of the relative error
  ! relative_denom : Denominator of the relative error
  ! converged      : Convergence check for outer loop
  logical   :: ok
  integer   :: istatus, isph, ibasis,  iteration
  real(dp)  :: inc, relative_num, relative_denom
  logical   :: converged

  !!
  !! Xadj_r         : Adjoint solution of Laplace equation
  !! Xadj_e         : Adjoint solution of HSP equation
  !! rhs_cosmo      : RHS corresponding to Laplace equation
  !! rhs_hsp        : RHS corresponding to HSP equation
  !! rhs_cosmo_init : Initial RHS for Laplace, psi_r in literature
  !! rhs_hsp_init   : Initial RHS for HSP, psi_e = 0 in literature
  !! X_r_k_1          : Solution of previous iterative step, holds Xadj_r_k_1
  !! X_e_k_1          : Solution of previous iterative step, holds Xadj_e_k_1
  real(dp), allocatable :: rhs_cosmo(:,:), rhs_hsp(:,:), &
                         & rhs_cosmo_init(:,:), rhs_hsp_init(:,:), &
                         & X_r_k_1(:,:), X_e_k_1(:,:)

  ! GMRES parameters
  integer :: info, n_iter
  real(dp) r_norm
  integer, parameter :: gmm = 20, gmj = 25
  real(dp), dimension(ddx_data % nsph*ddx_data % nbasis, 0:2*gmj+gmm+2-1) :: work

  ! Allocation
  allocate(rhs_cosmo(ddx_data % nbasis, ddx_data % nsph), &
           & rhs_hsp(ddx_data % nbasis, ddx_data % nsph), &
           & rhs_cosmo_init(ddx_data % nbasis, ddx_data % nsph), &
           & rhs_hsp_init(ddx_data % nbasis, ddx_data % nsph),&
           & X_r_k_1(ddx_data % nbasis, ddx_data % nsph),&
           & X_e_k_1(ddx_data % nbasis, ddx_data % nsph),&
           & stat = istatus)
  if(istatus .ne. 0) then
    write(*,*) 'Allocation failed in adjoint LPB!'
    stop
  end if
  ! We compute the adjoint solution first
  write(*,*) 'Solution of adjoint system'
  ! Set local variables
  n_iter = ddx_data % maxiter
  iteration = one
  inc = zero
  relative_num = zero
  relative_denom = zero
  X_r_k_1 = zero
  X_e_k_1 = zero
  ! Initial RHS
  ! rhs_cosmo_init = psi_r
  ! rhs_hsp_init   = psi_e (=0)
  rhs_cosmo_init = psi
  rhs_hsp_init = zero
  ! Updated RHS
  rhs_cosmo = rhs_cosmo_init
  rhs_hsp = rhs_hsp_init
  ! Initial Xadj_r and Xadj_e
  Xadj_r = zero
  Xadj_e = zero
  ! Logical variable for the first outer iteration
  first_out_iter = .true.
  converged = .false.
  ok = .false.
  ! Solve the adjoint system
  do while (.not.converged)
    ! Solve A*X_adj_r = psi_r
    ! Set the RHS to correct form by factoring with 4Pi/2l+1
    call convert_ddcosmo(ddx_data, 1, rhs_cosmo)
    call jacobi_diis(ddx_data, ddx_data % n, ddx_data % iprint, ddx_data % ndiis, &
                    & 4, ddx_data % tol, rhs_cosmo, Xadj_r, n_iter, &
                    & ok, lstarx, ldm1x, hnorm)

    ! Solve the HSP equation
    ! B*X_adj_e = psi_e
    ! For first iteration the rhs is zero for HSP equation. Hence, Xadj_e = 0
    work = zero
    Xadj_e = rhs_hsp
    if(iteration.ne.1) then
      call gmresr(ddx_data, .false., ddx_data % n, gmj, gmm, &
             & rhs_hsp, Xadj_e, work, tol_gmres,'rel', n_iter_gmres, r_norm, bstarx, info)

    endif

    ! Update the RHS
    ! |rhs_r| = |psi_r|-|C1* C1*||Xadj_r|
    ! |rhs_e| = |psi_e| |C2* C2*||Xadj_e|
    call update_rhs_adj(ddx_data, rhs_cosmo_init, rhs_hsp_init,&
                        & rhs_cosmo, rhs_hsp, Xadj_r, Xadj_e)
    ! Stopping Criteria.
    ! Checking the relative error of Xadj_r
    inc  = zero
    inc = dnrm2(ddx_data % n, Xadj_r + Xadj_e - X_r_k_1 - X_e_k_1, 1)/ &
        & dnrm2(ddx_data % n, Xadj_r + Xadj_e, 1)

    ! Store the previous step solution
    X_r_k_1 = Xadj_r
    X_e_k_1 = Xadj_e
    if ((iteration .gt. 1) .and. (inc.lt.ddx_data % tol)) then
      write(6,*) 'Reach tolerance.'
      converged = .true.
    end if
    write(6,*) 'Adjoint computation :', iteration, inc
    iteration = iteration + 1
    first_out_iter = .false.
  end do
  if (ddx_data % iprint .ge. 5) then
    call prtsph('Xadj_r', ddx_data % nbasis, ddx_data % lmax, &
          & ddx_data % nsph, 0, Xadj_r)
    call prtsph('Xadj_e', ddx_data % nbasis, ddx_data % lmax, &
          & ddx_data % nsph, 0, Xadj_e)
  end if
  ! Deallocation
  deallocate(rhs_cosmo, rhs_hsp, rhs_cosmo_init, &
           & rhs_hsp_init, X_r_k_1, X_e_k_1, stat = istatus)
  if(istatus .ne. 0) then
    write(*,*) 'Deallocation failed in adjoint LPB!'
    stop
  end if
  end subroutine ddx_lpb_adjoint

  !! Computation of Adjoint B, i.e., B*
  !> Apply adjoint single layer operator to spherical harmonics
  !! implementation is similar to lstarx in ddCOSMO
  !! Diagonal blocks are not counted here.
  subroutine bstarx(ddx_data, n, x, y)
  ! Inputs
  type(ddx_type), intent(in) :: ddx_data
  real(dp), intent(in)       :: x(ddx_data % nbasis, ddx_data % nsph)
  integer, intent(in)        :: n
  ! Output
  real(dp), intent(out)      :: y(ddx_data % nbasis, ddx_data % nsph)
  ! Local variables
  integer                    :: isph, igrid, istatus, l, ind
  real(dp), allocatable      :: xi(:,:), vplm(:), basloc(:), vcos(:), vsin(:)
  ! Allocate workspaces
  allocate(xi(ddx_data % ngrid, ddx_data % nsph), vplm(ddx_data % nbasis), &
        & basloc(ddx_data % nbasis), vcos(ddx_data % lmax+1), &
        & vsin(ddx_data % lmax+1), stat=istatus)
  if (istatus .ne. 0) then
      write(*, *) 'bstarx: allocation failed!'
      stop
  endif
  if (ddx_data % iprint .ge. 5) then
      call prtsph('X', ddx_data % nbasis, ddx_data % lmax, ddx_data % nsph, 0, &
          & x)
  end if
  ! Initalize
  y = zero
  !! Expand x over spherical harmonics
  ! Loop over spheres
  do isph = 1, ddx_data % nsph
      ! Loop over grid points
      do igrid = 1, ddx_data % ngrid
          xi(igrid, isph) = dot_product(x(:, isph), &
              & ddx_data % vgrid(:ddx_data % nbasis, igrid))
      end do
  end do
  !! Compute action
  ! Loop over spheres
  do isph = 1, ddx_data % nsph
      ! Compute NEGATIVE action of off-digonal blocks
      call adjrhs_lpb(ddx_data, isph, xi, y(:, isph), basloc, vplm, vcos, vsin)
      y(:, isph) = - y(:, isph)
      y(:,isph)  = y(:,isph) + x(:,isph)
  end do
  if (ddx_data % iprint .ge. 5) then
      call prtsph('B*X (off-diagonal)', ddx_data % nbasis, ddx_data % lmax, &
          & ddx_data % nsph, 0, y)
  end if
  !! Diagonal preconditioning for bstarx
  ! NOTE: Activate this if one is using GMRES solver
  !       Jacobi solver uses ldm1x for diagonals. Declare l and ind above.
  !do l = 0, ddx_data % lmax
  !  ind = l*l + l + 1
  !  y(ind-l:ind+l, :) = x(ind-l:ind+l, :) * (ddx_data % vscales(ind)**2)
  !end do
  ! Deallocate workspaces
  deallocate( xi, basloc, vplm, vcos, vsin , stat=istatus )
  if ( istatus.ne.0 ) then
      write(*,*) 'bstarx: allocation failed !'
      stop
  endif
  end subroutine bstarx

  !
  ! Taken from ddx_core routine adjrhs
  ! Called from bstarx
  ! Compute the Adjoint matix B*x
  !
  subroutine adjrhs_lpb( ddx_data, isph, xi, vlm, basloc, vplm, vcos, vsin )
  implicit none
  type(ddx_type), intent(in) :: ddx_data
  integer,  intent(in)    :: isph
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph), intent(in) :: xi
  real(dp), dimension((ddx_data % lmax+1)**2), intent(inout) :: vlm
  real(dp), dimension((ddx_data % lmax+1)**2), intent(inout) :: basloc, vplm
  real(dp), dimension(ddx_data % lmax+1), intent(inout) :: vcos, vsin

  integer :: ij, jsph, ig, l, ind, m
  real(dp)  :: vji(3), vvji, tji, sji(3), xji, oji, fac
  real(dp) :: rho, ctheta, stheta, cphi, sphi
  real(dp), dimension(ddx_data % nbasis) :: fac_hsp

  !loop over neighbors of i-sphere
  do ij = ddx_data % inl(isph), ddx_data % inl(isph+1)-1
    !j-sphere is neighbor
    jsph = ddx_data % nl(ij)
    !loop over integration points
    do ig = 1, ddx_data % ngrid
      !compute t_n^ji = | r_j + \rho_j s_n - r_i | / \rho_i
      vji  = ddx_data % csph(:,jsph) + ddx_data % rsph(jsph)* &
            & ddx_data % cgrid(:,ig) - ddx_data % csph(:,isph)
      vvji = sqrt(dot_product(vji,vji))
      tji  = vvji/ddx_data % rsph(isph)
      !point is INSIDE i-sphere (+ transition layer)
      if ( tji.lt.( one + (ddx_data % se+one)/two*ddx_data % eta ) ) then
        !compute s_n^ji
        sji = vji/vvji
        call ylmbas(sji, rho, ctheta, stheta, cphi, &
                      & sphi, ddx_data % lmax, &
                      & ddx_data % vscales, basloc, &
                      & vplm, vcos, vsin)
        call inthsp_adj(ddx_data, vvji, ddx_data % rsph(isph), isph, basloc, fac_hsp)
        !compute \chi( t_n^ji )
        xji = fsw( tji, ddx_data % se, ddx_data % eta )
        !compute W_n^ji
        if ( ddx_data % fi(ig,jsph).gt.one ) then
          oji = xji/ ddx_data % fi(ig,jsph)
        else
          oji = xji
        endif
        !compute w_n * xi(n,j) * W_n^ji
        fac = ddx_data % wgrid(ig) * xi(ig,jsph) * oji
        !loop over l
        do l = 0, ddx_data % lmax
          ind  = l*l + l + 1
          !loop over m
            do m = -l,l
              vlm(ind+m) = vlm(ind+m) + fac*fac_hsp(ind+m)
            enddo
        enddo
      endif
    enddo
  enddo
  end subroutine adjrhs_lpb
  
  !
  ! Intermediate calculation in adjrhs_lpb subroutine
  ! @param[in]  rijn    : Radius of sphers x_ijn
  ! @param[in]  ri      : Radius of sphers x_i
  ! @param[in]  isph    : Index of sphere
  ! @param[in]  basloc  : Spherical Harmonic
  ! @param[out] fac_hsp : Return bessel function ratio multiplied by 
  !                       the spherical harmonic Y_l'm'. Array of size nylm
  !
  subroutine inthsp_adj(ddx_data, rjin, rj, jsph, basloc, fac_hsp)
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  integer, intent(in) :: jsph
  real(dp), intent(in) :: rjin, rj
  real(dp), dimension(ddx_data % nbasis), intent(in) :: basloc
  real(dp), dimension(ddx_data % nbasis), intent(inout) :: fac_hsp
  real(dp), dimension(0:ddx_data % lmax) :: SI_rjin, DI_rjin
  integer :: l, m, ind, NM

  SI_rjin = 0
  DI_rjin = 0
  fac_hsp = 0

  ! Computation of modified spherical Bessel function values      
  !call SPHI_bessel(ddx_data % lmax,rjin*ddx_data % kappa,NM,SI_rjin,DI_rjin)
  call modified_spherical_bessel_first_kind(ddx_data % lmax, rjin*ddx_data % kappa, SI_rjin, DI_rjin)
  
  do l = 0, ddx_data % lmax
    do  m = -l, l
      ind = l*l + l + 1 + m
      fac_hsp(ind) = SI_rjin(l)/SI_ri(l,jsph)*basloc(ind)
    end do
  end do
  endsubroutine inthsp_adj
  
  !
  ! Update the RHS in outer iteration for adjoint system
  ! @param[in] rhs_r_init : psi_r
  ! @param[in] rhs_e_init : psi_e = 0
  ! @param[in, out] rhs_r : -C_1*\times Xadj_r^(k-1) - C_1*\times Xadj_e^(k-1)
  !                         + psi_r
  ! @param[in, out] rhs_e : -C_2*\times Xadj_r^(k-1) - C_2*\times Xadj_e^(k-1)
  ! @param[in] Xadj_r     : Xadj_r^(k-1)
  ! @param[in] Xadj_e     : Xadj_e^(k-1)
  !
  subroutine update_rhs_adj(ddx_data, rhs_r_init, rhs_e_init, rhs_r, &
      & rhs_e, Xadj_r, Xadj_e)
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(in) :: rhs_r_init, &
                                                                       & rhs_e_init
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(in) :: Xadj_r, Xadj_e
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(out) :: rhs_r, rhs_e
  ! Local Variables
  ! rhs_r_adj : C_1*(Xadj_r+Xadj_e)
  ! rhs_e_adj : C_2*(Xadj_r+Xadj_e)
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph) :: rhs_adj
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: Xadj_sgrid
  ! isph    : Index for the sphere
  ! igrid   : Index for the grid points
  ! icav    : Index for the external grid points
  ! ibasis  : Index for the basis
  ! ibasis0 : Index for the fixed basis (nbasis0)
  ! l       : Index for lmax
  ! m       : Index for m:-l,...,l
  ! ind     : l^2+l+m+1
  integer :: isph, igrid, icav, ibasis, ibasis0, l, m, ind, jsph
  ! epsilon_ratio : epsilon_1/epsilon_2
  real(dp) :: epsilon_ratio
  ! val_1, val_2  : Intermediate summations variable
  real(dp) :: val

  epsilon_ratio = epsp/ddx_data % eps

  ! NOTE: These remain constant through the outer iteration and hence needs to be computed
  !       once.
  if(first_out_iter) then
    ! Compute
    ! diff_ep_adj = Pchi * coefY
    ! Summation over l0, m0
    diff_ep_adj = zero
    do icav = 1, ddx_data % ncav
      do ibasis = 1, ddx_data % nbasis
        do isph = 1, ddx_data % nsph
          val = zero
          do ibasis0 = 1, nbasis0
            val = val + Pchi(ibasis,ibasis0,isph)*coefY(icav,ibasis0,isph)
          end do
          diff_ep_adj(icav, ibasis, isph) = val
        end do
      end do
    end do
  endif

  ! Call dgemm to integrate the adjoint solution on the grid points
  ! Summation over l' and m'
  call dgemm('T', 'N', ddx_data % ngrid, ddx_data % nsph, &
            & ddx_data % nbasis, one, ddx_data % vgrid, ddx_data % vgrid_nbasis, &
            & Xadj_r + Xadj_e , ddx_data % nbasis, zero, Xadj_sgrid, &
            & ddx_data % ngrid)


  rhs_adj = zero
  do isph = 1, ddx_data % nsph
    do ibasis = 1, ddx_data % nbasis
      icav = zero
      !Summation over j and n
      do jsph = 1, ddx_data % nsph
        do igrid = 1, ddx_data % ngrid
          if (ddx_data % ui(igrid,jsph).gt.zero) then
            icav = icav + 1
            rhs_adj(ibasis, isph) = rhs_adj(ibasis, isph) + &
                        & ddx_data % wgrid(igrid)*&
                        & ddx_data % ui(igrid, jsph)*&
                        & Xadj_sgrid(igrid, jsph)*&
                        & diff_ep_adj(icav, ibasis, isph)
          end if
        end do
      end do
    end do
  end do

  do isph = 1, ddx_data % nsph
    do l = 0, ddx_data % lmax
      do m = -l, l
        ind = l**2 + l + m + 1
        rhs_r(ind, isph) = rhs_r_init(ind, isph) - &
                        & (epsilon_ratio*l*rhs_adj(ind, isph))/ddx_data % rsph(isph)
        rhs_e(ind, isph) = rhs_e_init(ind, isph) + termimat(l,isph)*rhs_adj(ind, isph)
      end do
    end do
  end do

  return
  end subroutine update_rhs_adj
  
  !
  ! Subroutine to compute K^A counterpart for the HSP equation. Similar to fdoka.
  ! @param[in]  ddx_data  : Data type
  ! @param[in]  isph      : Index of sphere
  ! @param[in]  Xe        : Solution vector Xe
  ! @param[in]  Xadj_e    : Adjoint solution on evaluated on grid points Xadj_e_sgrid
  ! @param[in]  basloc    : Spherical harmonics Y_lm
  ! @param[in]  dbasloc   : Derivative of spherical harmonics \nabla^i(Y_lm)
  ! @param[in]  vplm      : Argument to call ylmbas
  ! @param[in]  vcos      : Argument to call ylmbas
  ! @param[in]  vsin      : Argument to call ylmbas
  ! @param[out] force_e   : Force of adjoint part
  subroutine fdoka_b_xe(ddx_data, isph, Xe, Xadj_e, basloc, dbasloc, &
                       & vplm, vcos, vsin, force_e)
  use bessel
  implicit none
  type(ddx_type), intent(in) :: ddx_data
  integer,                         intent(in)    :: isph
  real(dp),  dimension(ddx_data % nbasis, ddx_data % nsph), intent(in)   :: Xe
  real(dp),  dimension(ddx_data % ngrid),       intent(in)    :: Xadj_e
  real(dp),  dimension(ddx_data % nbasis),      intent(inout) :: basloc, vplm
  real(dp),  dimension(3, ddx_data % nbasis),    intent(inout) :: dbasloc
  real(dp),  dimension(ddx_data % lmax+1),      intent(inout) :: vcos, vsin
  real(dp),  dimension(3),           intent(inout) :: force_e

  ! Local Variables
  ! igrid  : Index of grid point
  ! ineigh : Index over Row space of neighbors
  ! jsph   : Index of neighbor sphere
  ! l     : Index for l, l:0,...,lmax
  ! m     : Index for m, m:-l,...,l
  ! ind    : l*l+l+1
  ! NM     : Argument for calling Bessel functions
  integer :: igrid, ineigh, jsph, l, ind, m, NM
  ! SI_rijn : Besssel function of first kind for rijn
  ! DI_rijn : Derivative of Besssel function of first kind for rijn
  real(dp), dimension(0:ddx_data % lmax) :: SI_rijn
  real(dp), dimension(0:ddx_data % lmax) :: DI_rijn
  ! rijn   : r_j*r_1^j(x_i^n) = |x_i^n-x_j|
  ! tij    : r_1^j(x_i^n)
  ! beta   : Eq.(53) Stamm.etal.18
  ! tlow   : Lower bound for switch region
  ! thigh  : Upper bound for switch region
  ! xij    : chi_j(x_i^n)
  ! oij    : omega^\eta_ijn
  ! f1     : First factor in alpha computation
  ! f2     : Second factor in alpha computation
  real(dp)  :: rijn, tij, beta, tlow, thigh, xij, oij, f1, f2, f3
  ! vij   : x_i^n-x_j
  ! sij   : e^j(x_i^n)
  ! alpha : Eq.(52) Stamm.etal.18
  ! va    : Eq.(54) Stamm.etal.18
  real(dp)  :: vij(3), sij(3), alpha(3), va(3), rj
  real(dp), external :: dnrm2
  
  SI_rijn = 0
  DI_rijn = 0
  
  tlow  = one - pt5*(one - ddx_data % se)*ddx_data % eta
  thigh = one + pt5*(one + ddx_data % se)*ddx_data % eta

  ! Loop over grid points
  do igrid = 1, ddx_data % ngrid
    va = zero
    do ineigh = ddx_data % inl(isph), ddx_data % inl(isph+1) - 1
      jsph = ddx_data % nl(ineigh)
      vij  = ddx_data % csph(:,isph) + &
            & ddx_data % rsph(isph)*ddx_data % cgrid(:,igrid) - &
            & ddx_data % csph(:,jsph)
      rijn = dnrm2(3, vij, 1)
      tij  = rijn/ddx_data % rsph(jsph)
      rj = ddx_data % rsph(jsph)

      if (tij.ge.thigh) cycle
      ! Computation of modified spherical Bessel function values      
      call modified_spherical_bessel_first_kind(ddx_data % lmax, rijn*ddx_data % kappa, SI_rijn, DI_rijn)

      sij  = vij/rijn
      !call dbasis(sij,basloc,dbasloc,vplm,vcos,vsin)
      call dbasis(ddx_data, sij, basloc, dbasloc, vplm, vcos, vsin)
      alpha  = zero
      do l = 0, ddx_data % lmax
        ind = l*l + l + 1
        f1 = (DI_rijn(l)*ddx_data % kappa)/SI_ri(l,jsph);
        f2 = SI_rijn(l)/SI_ri(l, jsph)
        do m = -l, l
          alpha(:) = alpha(:) + (f1*sij(:)*basloc(ind+m) + &
                    & (f2/rijn)*dbasloc(:,ind+m))*Xe(ind+m,jsph)
        end do
      end do
      beta = compute_beta(ddx_data, SI_rijn, rijn, jsph, Xe(:,jsph),basloc)
      xij = fsw(tij, ddx_data % se, ddx_data % eta)
      if (ddx_data % fi(igrid,isph).gt.one) then
        oij = xij/ddx_data % fi(igrid,isph)
        f2  = -oij/ddx_data % fi(igrid,isph)
      else
        oij = xij
        f2  = zero
      end if
      f1 = oij
      va(:) = va(:) + f1*alpha(:) + beta*f2*ddx_data % zi(:,igrid,isph)
      if (tij .gt. tlow) then
        f3 = beta*dfsw(tij,ddx_data % se,ddx_data % eta)/ddx_data % rsph(jsph)
        if (ddx_data % fi(igrid,isph).gt.one) f3 = f3/ddx_data % fi(igrid,isph)
        va(:) = va(:) + f3*sij(:)
      end if
    end do
  force_e = force_e - ddx_data % wgrid(igrid)*Xadj_e(igrid)*va(:)
  end do
  return
  end subroutine fdoka_b_xe
  
  !
  ! Subroutine to compute K^A+K^C counterpart for the HSP equation. Similar to fdokb.
  ! @param[in]  ddx_data  : Data type
  ! @param[in]  isph      : Index of sphere
  ! @param[in]  Xe        : Solution vector Xe
  ! @param[in]  Xadj_e    : Adjoint solution on evaluated on grid points Xadj_e_sgrid
  ! @param[in]  basloc    : Spherical harmonics Y_lm
  ! @param[in]  dbasloc   : Derivative of spherical harmonics \nabla^i(Y_lm)
  ! @param[in]  vplm      : Argument to call ylmbas
  ! @param[in]  vcos      : Argument to call ylmbas
  ! @param[in]  vsin      : Argument to call ylmbas
  ! @param[out] force_e   : Force of adjoint part
  subroutine fdokb_b_xe(ddx_data, isph, Xe, Xadj_e, basloc, dbasloc, &
                        & vplm, vcos, vsin, force_e)
  use bessel
  implicit none
  type(ddx_type), intent(in) :: ddx_data
  integer,                         intent(in)    :: isph
  real(dp),  dimension(ddx_data % nbasis, ddx_data % nsph), intent(in)    :: Xe
  real(dp),  dimension(ddx_data % ngrid, ddx_data % nsph),  intent(in)    :: Xadj_e
  real(dp),  dimension(ddx_data % nbasis),      intent(inout) :: basloc,  vplm
  real(dp),  dimension(3, ddx_data % nbasis),   intent(inout) :: dbasloc
  real(dp),  dimension(ddx_data % lmax+1),      intent(inout) :: vcos, vsin
  real(dp),  dimension(3),           intent(inout) :: force_e

  ! Local Variables
  ! igrid  : Index of grid points
  ! jsph   : Index of jth sphere
  ! ksph   : Index of kth sphere
  ! ineigh : Row pointer over ith row
  ! l      : Index for l, l:0,...,lmax
  ! m      : Index for m, m:-l0,...,l0
  ! ind    : l*l+l+1
  ! jk     : Row pointer over kth row
  ! NM     : Argument for calling Bessel functions
  integer :: igrid, jsph, ksph, ineigh, l, m, ind, jk,  NM
  ! SI_rjin : Besssel function of first kind for rijn
  ! DI_rjin : Derivative of Besssel function of first kind for rijn
  ! SI_rjkn : Besssel function of first kind for rjkn
  ! DI_rjkn : Derivative of Besssel function of first kind for rjkn
  real(dp), dimension(0:ddx_data % lmax) :: SI_rjin, SI_rjkn
  real(dp), dimension(0:ddx_data % lmax) :: DI_rjin, DI_rjkn

  logical :: proc
  ! rjin    : r_i*r_1^i(x_j^n) = |x_j^n-x_i|
  ! tji     : r_1^i(x_j^n)
  ! xji     : chi_i(x_j^n)
  ! oji     : omega^\eta_jin
  ! fac     : \delta_fj_n*\omega^\eta_ji
  ! f1      : First factor in alpha computation
  ! f2      : Second factor in alpha computation
  ! beta_ji : Eq.(57) Stamm.etal.18
  ! dj      : Before Eq.(10) Stamm.etal.18
  ! tlow    : Lower bound for switch region
  ! thigh   : Upper bound for switch region
  real(dp)  :: rjin, tji, xji, oji, fac, f1, f2, beta_ji, dj, tlow, thigh
  ! beta_jk : Eq.(58) Stamm.etal.18
  ! rjkn    : r_k*r_1^k(x_j^n) = |x_j^n-x_k|
  ! tjk     : r_1^k(x_j^n)
  ! xjk     : chi_k(x_j^n)
  real(dp)  :: b, beta_jk, g1, g2, rjkn, tjk, xjk
  ! vji   : x_j^n-x_i
  ! sji   : e^i(x_j^n)
  ! vjk   : x_j^n-x_k
  ! sjk   : e^k(x_j^n)
  ! alpha : Eq.(56) Stamm.etal.18
  ! vb    : Eq.(60) Stamm.etal.18
  ! vc    : Eq.(59) Stamm.etal.18
  real(dp)  :: vji(3), sji(3), vjk(3), sjk(3), alpha(3), vb(3), vc(3)
  ! rho    : Argument for ylmbas
  ! ctheta : Argument for ylmbas
  ! stheta : Argument for ylmbas
  ! cphi   : Argument for ylmbas
  ! sphi   : Argument for ylmbas
  real(dp) :: rho, ctheta, stheta, cphi, sphi, ri, arg_bessel

  real(dp), external :: dnrm2
  
  SI_rjin = 0
  DI_rjin = 0
  SI_rjkn = 0
  DI_rjkn = 0

  tlow  = one - pt5*(one - ddx_data % se)*ddx_data % eta
  thigh = one + pt5*(one + ddx_data % se)*ddx_data % eta

  do igrid = 1, ddx_data % ngrid
    vb = zero
    vc = zero
    do ineigh = ddx_data % inl(isph), ddx_data % inl(isph+1) - 1
      jsph = ddx_data % nl(ineigh)
      vji  = ddx_data % csph(:,jsph) + &
              & ddx_data % rsph(jsph)*ddx_data % cgrid(:,igrid) - &
              & ddx_data % csph(:,isph)
      rjin = dnrm2(3, vji, 1)
      ri = ddx_data % rsph(isph)
      tji  = rjin/ri

      if (tji.gt.thigh) cycle

      call modified_spherical_bessel_first_kind(ddx_data % lmax, rjin*ddx_data % kappa, SI_rjin, DI_rjin)
      
      sji  = vji/rjin
      call dbasis(ddx_data, sji, basloc, dbasloc, vplm, vcos, vsin)
      alpha = zero
      do l = 0, ddx_data % lmax
        ind = l*l + l + 1
        f1 = (DI_rjin(l)*ddx_data % kappa)/SI_ri(l,isph);
        f2 = SI_rjin(l)/SI_ri(l,isph)

        do m = -l, l
          alpha = alpha + (f1*sji*basloc(ind+m) + &
                 & (f2/rjin)*dbasloc(:,ind+m))*Xe(ind+m,isph)
        end do
      end do
      xji = fsw(tji,ddx_data % se,ddx_data % eta)
      if (ddx_data % fi(igrid,jsph).gt.one) then
        oji = xji/ddx_data % fi(igrid,jsph)
      else
        oji = xji
      end if
      f1 = oji
      vb = vb + f1*alpha*Xadj_e(igrid,jsph)
      if (tji .gt. tlow) then
        ! Compute beta_jin, i.e., Eq.(57) Stamm.etal.18
        beta_ji = compute_beta(ddx_data, SI_rjin, rjin, isph, Xe(:,isph), basloc)
        if (ddx_data % fi(igrid,jsph) .gt. one) then
          dj  = one/ddx_data % fi(igrid,jsph)
          fac = dj*xji
          proc = .false.
          b    = zero
          do jk = ddx_data % inl(jsph), ddx_data % inl(jsph+1) - 1
            ksph = ddx_data % nl(jk)
            vjk  = ddx_data % csph(:,jsph) + &
                 & ddx_data % rsph(jsph)*ddx_data % cgrid(:,igrid) - &
                 & ddx_data % csph(:,ksph)
            rjkn = dnrm2(3, vjk, 1)
            tjk  = rjkn/ddx_data % rsph(ksph)
            ! Computation of modified spherical Bessel function values      
            call modified_spherical_bessel_first_kind(ddx_data % lmax, rjkn*ddx_data % kappa, SI_rjkn, DI_rjkn)

            if (ksph.ne.isph) then
              if (tjk .le. thigh) then
              proc = .true.
              sjk  = vjk/rjkn
              call ylmbas(sjk, rho, ctheta, stheta, cphi, sphi, &
                  & ddx_data % lmax, ddx_data % vscales, basloc, vplm, &
                  & vcos, vsin)
              beta_jk  = compute_beta(ddx_data, SI_rjkn, rjkn, ksph, Xe(:,ksph), basloc)
              xjk = fsw(tjk, ddx_data % se, ddx_data % eta)
              b   = b + beta_jk*xjk
              end if
            end if
          end do
          if (proc) then
            g1 = dj*dj*dfsw(tji,ddx_data % se,ddx_data % eta)/ddx_data % rsph(isph)
            g2 = g1*Xadj_e(igrid,jsph)*b
            vc = vc + g2*sji
          end if
        else
          dj  = one
          fac = zero
        end if
        f2 = (one-fac)*dj*dfsw(tji,ddx_data % se,ddx_data % eta)/ddx_data % rsph(isph)
        vb = vb + f2*Xadj_e(igrid,jsph)*beta_ji*sji
      end if 
    end do
    force_e = force_e + ddx_data % wgrid(igrid)*(vb - vc)
    end do
  return
  end subroutine fdokb_b_xe
  
  real(dp) function compute_beta(ddx_data, SI_rijn, rijn, jsph, Xe, basloc)
  implicit none
  type(ddx_type) :: ddx_data
  real(dp), dimension(0:ddx_data % lmax), intent(in) :: SI_rijn
  real(dp), dimension(ddx_data % nbasis), intent(in) :: basloc
  real(dp), dimension(ddx_data % nbasis), intent(in)   :: Xe
  integer, intent(in) :: jsph
  real(dp), intent(in) :: rijn

  integer :: l, m, ind
  real(dp)  :: ss, fac
  ss = zero

  ! loop over l
  do l = 0, ddx_data % lmax
    do m = -l, l
      ind = l*l + l + m + 1
      fac = SI_rijn(l)/SI_ri(l,jsph)
      ss = ss + fac*basloc(ind)*Xe(ind)
    end do
  end do
     
  compute_beta = ss
  end function compute_beta
  
  !
  ! Subroutine to compute the derivative of U_i(x_in) and \bf{k}^j_l0(x_in)Y^j_l0m0(x_in)
  ! fdouky : Force Derivative of U_i^e(x_in), k_l0, and Y_l0m0
  ! @param[in] ddx_data     : Data Type
  ! @param[in] ksph         : Derivative with respect to x_k
  ! @param[in] Xr           : Solution of the Laplace problem
  ! @param[in] Xe           : Solution of the HSP problem
  ! @param[in] Xadj_r_sgrid : Solution of the Adjoint Laplace problem evaluated at the grid
  ! @param[in] Xadj_e_sgrid : Solution of the Adjoint HSP problem evaluated at the grid
  ! @param[in] Xadj_r       : Adjoint solution of the Laplace problem
  ! @param[in] Xadj_e       : Adjoint solution of the HSP problem
  ! @param[inout] force     : Force
  ! @param[out] diff_re     : epsilon_1/epsilon_2 * l'/r_j[Xr]_jl'm' 
  !                         - (i'_l'(r_j)/i_l'(r_j))[Xe]_jl'm'
  subroutine fdouky(ddx_data, ksph, Xr, Xe, &
                          & Xadj_r_sgrid, Xadj_e_sgrid, &
                          & Xadj_r, Xadj_e, &
                          & force, &
                          & diff_re)
  implicit none
  type(ddx_type), intent(in) :: ddx_data
  integer, intent(in) :: ksph
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: Xr, Xe
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph), intent(in) :: Xadj_r_sgrid,&
                                                                        & Xadj_e_sgrid
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: Xadj_r, Xadj_e
  real(dp), dimension(3), intent(inout) :: force
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(out) :: diff_re
  real(dp), external :: dnrm2
  ! Local variable
  ! isph  : Index for sphere i
  ! jsph  : Index for sphere j
  ! igrid : Index for grid point
  ! l     : l=0, lmax
  ! m     : -l,..,l
  ! ind   : l^2+l+1
  ! l0    : l0=0, lmax0
  ! m0    : -l0, l0
  ! ind0  : l0^2+l0+1
  ! kep   : Index for external grid point
  integer :: isph, jsph, igrid, l, m, ind, l0, m0, ind0, kep
  ! vij      : x_i^n-x_k
  ! sij      : e^j(x_i^n)
  ! val_dim3 : Intermediate value array of dimension 3
  real(dp), dimension(3) :: sij, vij, val_dim3
  ! val   : Intermediate variable to compute diff_ep
  ! f1    : Intermediate variable for derivative of coefY_der
  ! f2    : Intermediate variable for derivative of coefY_der
  real(dp) :: val, f1, f2
  ! phi_in : sum_{j=1}^N diff0_j * coefY_j
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: phi_in
  ! diff_ep_dim3 : 3 dimensional couterpart of diff_ep
  real(dp), dimension(3, ddx_data % ncav) :: diff_ep_dim3
  ! sum_dim3 : Storage of sum
  real(dp), dimension(3, ddx_data % nbasis, ddx_data % nsph) :: sum_dim3
  ! coefY_der : Derivative of k_l0 and Y_l0m0
  real(dp), dimension(3, ddx_data % ncav, nbasis0, ddx_data % nsph) :: coefY_der
  ! Debug purpose
  ! These variables can be taken from the subroutine update_rhs
  ! diff0       : dot_product([PU_j]_l0m0^l'm', l'/r_j[Xr]_jl'm' -
  !                        (i'_l'(r_j)/i_l'(r_j))[Xe]_jl'm')

  real(dp), dimension(nbasis0, ddx_data % nsph) :: diff0
  real(dp) :: termi, termk, rijn
  ! basloc : Y_lm(s_n)
  ! vplm   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % nbasis):: basloc, vplm
  ! dbasloc : Derivative of Y_lm(s_n)
  real(dp),  dimension(3, ddx_data % nbasis):: dbasloc
  ! vcos   : Argument to call ylmbas
  ! vsin   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % lmax+1):: vcos, vsin
  ! SK_rijn : Besssel function of first kind for rijn
  ! DK_rijn : Derivative of Besssel function of first kind for rijn
  real(dp), dimension(0:ddx_data % lmax) :: SK_rijn, DK_rijn


  ! Setting initial values to zero
  SK_rijn = zero
  DK_rijn = zero
  coefY_der = zero

  !Compute coefY = C_ik*Y_lm(x_in)*\bar(k_l0^j(x_in))
  kep = 0
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if (ddx_data % ui(igrid,isph).gt.zero) then
        kep = kep + 1
        ! Loop to compute Sijn
        do jsph = 1, ddx_data % nsph
          vij  = ddx_data % csph(:,isph) + &
                & ddx_data % rsph(isph)*ddx_data % cgrid(:,igrid) - &
                & ddx_data % csph(:,jsph)
          rijn = sqrt(dot_product(vij,vij))
          sij = vij/rijn

          call modified_spherical_bessel_second_kind(lmax0, &
                                                     & rijn*ddx_data % kappa, &
                                                     & SK_rijn, DK_rijn)
          call dbasis(ddx_data, sij, basloc, dbasloc, vplm, vcos, vsin)

          do l0 = 0,lmax0
            f1 = (DK_rijn(l0)*ddx_data % kappa)/SK_ri(l0,jsph)
            f2 = SK_rijn(l0)/SK_ri(l0,jsph)
            do m0 = -l0, l0
              ind0 = l0**2 + l0 + m0 + 1
              ! coefY_der : Derivative of Bessel function and spherical harmonic
              ! Non-Diagonal entries
              if ((ksph .eq. isph) .and. (isph .ne. jsph)) then
                coefY_der(:,kep,ind0,jsph) = C_ik(l0,jsph)*(f1*sij*basloc(ind0) + &
                                             & (f2/rijn)*dbasloc(:,ind0))
              elseif ((ksph .eq. jsph) .and. (isph .ne. jsph)) then
                coefY_der(:,kep,ind0,jsph) = -C_ik(l0,jsph)*(f1*sij*basloc(ind0)+ &
                                             & (f2/rijn)*dbasloc(:,ind0))
              else
                coefY_der(:,kep,ind0,jsph) = zero
              endif
            end do ! End of loop m0
          end do ! End of l0
        end do ! End of loop jsph
      end if
    end do ! End of loop igrid
  end do ! End of loop isph

  diff_re = zero
  ! Compute l'/r_j[Xr]_jl'm' -(i'_l'(r_j)/i_l'(r_j))[Xe]_jl'm'
  do jsph = 1, ddx_data % nsph
    do l = 0, ddx_data % lmax
      do m = -l,l
        ind = l**2 + l + m + 1
        diff_re(ind,jsph) = (epsp/ddx_data % eps)*(l/ddx_data % rsph(jsph)) * &
              & Xr(ind,jsph) - termimat(l,jsph)*Xe(ind,jsph)
      end do
    end do
  end do

  ! diff0 = Pchi * diff_re, linear scaling
  diff0 = zero
  do jsph = 1, ddx_data % nsph
    do ind0 = 1, nbasis0
      diff0(ind0, jsph) = dot_product(diff_re(:,jsph), &
          & Pchi(:,ind0, jsph))
    end do
  end do
  ! phi_in = diff0 * coefY
  ! Here, summation over j takes place
  phi_in = zero
  kep = zero
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if(ddx_data % ui(igrid, isph) .gt. zero) then
        ! Extrenal grid point
        kep = kep + 1
        val = zero
        val_dim3(:) = zero
        do jsph = 1, ddx_data % nsph 
          do ind0 = 1, nbasis0
            val = val + diff0(ind0,jsph)*coefY(kep,ind0,jsph)
            val_dim3(:) = val_dim3(:) + diff0(ind0,jsph)*coefY_der(:, kep, ind0, jsph)
          end do
        end do
        diff_ep_dim3(:, kep) = val_dim3(:)
      end if
    phi_in(igrid, isph) = val
    end do
  end do
  ! Computation of derivative of U_i^e(x_in)
  call fdoga(ddx_data, ksph, Xadj_r_sgrid, phi_in, force)
  call fdoga(ddx_data, ksph, Xadj_e_sgrid, phi_in, force)

  sum_dim3 = zero
  kep = zero
  do isph = 1, ddx_data % nsph
    do igrid =1, ddx_data % ngrid
      if(ddx_data % ui(igrid, isph) .gt. zero) then
        kep = kep + 1
        do ind = 1, ddx_data % nbasis
          sum_dim3(:,ind,isph) = sum_dim3(:,ind,isph) + &
                                & coefvec(igrid, ind, isph)*diff_ep_dim3(:,kep)
        end do
      end if
    end do
  end do
  ! Computation of derivative of \bf(k)_j^l0(x_in)\times Y^j_l0m0(x_in)
  do isph = 1, ddx_data % nsph
    do ind = 1, ddx_data % nbasis
      force = force + sum_dim3(:, ind, isph)*(Xadj_r(ind, isph) + &
             & Xadj_e(ind, isph))
    end do
  end do

  end subroutine fdouky

  !
  ! Subroutine to compute the derivative of U_i(x_in) and \bf{k}^j_l0(x_in)Y^j_l0m0(x_in)
  ! in F0
  ! fdouky_f0 : Force Derivative of U_i^e(x_in), k_l0, and Y_l0m0 in F0
  ! @param[in] ddx_data     : Data Type
  ! @param[in] ksph         : Derivative with respect to x_k
  ! @param[in] sol_sgrid    : Solution of the Adjoint problem evaluated at the grid
  ! @param[in] sol_adj      : Adjoint solution
  ! @param[in] gradpsi      : Gradient of Psi_0
  ! @param[inout] force     : Force
  subroutine fdouky_f0(ddx_data, ksph,&
                          & sol_adj, sol_sgrid, gradpsi, &
                          & force)
  implicit none
  type(ddx_type), intent(in) :: ddx_data
  integer, intent(in) :: ksph
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: sol_adj
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph), intent(in) :: sol_sgrid
  real(dp), dimension(3, ddx_data % ncav), intent(in) :: gradpsi
  real(dp), dimension(3), intent(inout) :: force
  real(dp), external :: dnrm2
  ! Local variable
  ! isph  : Index for sphere i
  ! jsph  : Index for sphere j
  ! igrid : Index for grid point
  ! l     : l=0, lmax
  ! m     : -l,..,l
  ! ind   : l^2+l+1
  ! l0    : l0=0, lmax0
  ! m0    : -l0, l0
  ! ind0  : l0^2+l0+1
  ! kep   : Index for external grid point
  integer :: isph, jsph, igrid, l, m, ind, l0, m0, ind0, icav
  ! vij      : x_i^n-x_k
  ! sij      : e^j(x_i^n)
  ! val_dim3 : Intermediate value array of dimension 3
  real(dp), dimension(3) :: sij, vij, val_dim3
  ! val     : Intermediate variable to compute diff_ep
  ! f1      : Intermediate variable for derivative of coefY_der
  ! f2      : Intermediate variable for derivative of coefY_der
  ! nderpsi : Derivative of psi on grid points
  real(dp) :: val, f1, f2, nderpsi, sum_int
  ! phi_in : sum_{j=1}^N diff0_j * coefY_j
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: phi_in
  ! diff_ep_dim3 : 3 dimensional couterpart of diff_ep
  real(dp), dimension(3, ddx_data % ncav) :: diff_ep_dim3
  ! sum_dim3 : Storage of sum
  real(dp), dimension(3, ddx_data % nbasis, ddx_data % nsph) :: sum_dim3
  ! coefY_der : Derivative of k_l0 and Y_l0m0
  real(dp), dimension(3, ddx_data % ncav, nbasis0, ddx_data % nsph) :: coefY_der
  ! Debug purpose
  ! These variables can be taken from the subroutine update_rhs
  ! diff0       : dot_product([PU_j]_l0m0^l'm', l'/r_j[Xr]_jl'm' -
  !                        (i'_l'(r_j)/i_l'(r_j))[Xe]_jl'm')

  real(dp), dimension(nbasis0, ddx_data % nsph) :: diff0
  real(dp) :: termi, termk, rijn
  ! basloc : Y_lm(s_n)
  ! vplm   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % nbasis):: basloc, vplm
  ! dbasloc : Derivative of Y_lm(s_n)
  real(dp),  dimension(3, ddx_data % nbasis):: dbasloc
  ! vcos   : Argument to call ylmbas
  ! vsin   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % lmax+1):: vcos, vsin
  ! SK_rijn : Besssel function of first kind for rijn
  ! DK_rijn : Derivative of Besssel function of first kind for rijn
  real(dp), dimension(0:ddx_data % lmax) :: SK_rijn, DK_rijn
  ! sum_Sjin : \sum_j [S]_{jin} Eq.~(97) [QSM20.SISC]
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: sum_Sjin
  ! c0 : \sum_{n=1}^N_g w_n U_j^{x_nj}\partial_n psi_0(x_nj)Y_{l0m0}(s_n)
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph) :: c0_d

  ! Setting initial values to zero
  SK_rijn = zero
  DK_rijn = zero
  coefY_der = zero
  c0_d = zero

  icav = zero
  do isph = 1, ddx_data % nsph
    do igrid= 1, ddx_data % ngrid
      if ( ddx_data % ui(igrid,isph) .gt. zero ) then
        icav = icav + 1
        nderpsi = dot_product( gradpsi(:,icav),ddx_data % cgrid(:,igrid) )
        c0_d(:, isph) = c0_d(:,isph) + &
                     & ddx_data % wgrid(igrid)* &
                     & ddx_data % ui(igrid,isph)*&
                     & nderpsi* &
                     & ddx_data % vgrid(:,igrid)
      end if
    end do
  end do

  ! Compute [S]_{jin}
  icav = 0
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if (ddx_data % ui(igrid,isph).gt.zero) then
        icav = icav + 1
        sum_int = zero
        ! Loop to compute Sijn
        do jsph = 1, ddx_data % nsph
          vij  = ddx_data % csph(:,isph) + &
                & ddx_data % rsph(isph)*ddx_data % cgrid(:,igrid) - &
                & ddx_data % csph(:,jsph)
          rijn = sqrt(dot_product(vij,vij))
          sij = vij/rijn

          call modified_spherical_bessel_second_kind(lmax0, &
                                                     & rijn*ddx_data % kappa, &
                                                     & SK_rijn, DK_rijn)
          call dbasis(ddx_data, sij, basloc, dbasloc, vplm, vcos, vsin)

          do l0 = 0,lmax0
            f1 = (DK_rijn(l0)*ddx_data % kappa)/SK_ri(l0,jsph)
            f2 = SK_rijn(l0)/SK_ri(l0,jsph)
            do m0 = -l0, l0
              ind0 = l0**2 + l0 + m0 + 1
              sum_int = sum_int + c0_d(ind0,jsph)*coefY(icav, ind0, jsph)
              ! coefY_der : Derivative of Bessel function and spherical harmonic
              ! Non-Diagonal entries
              if ((ksph .eq. isph) .and. (isph .ne. jsph)) then
                coefY_der(:,icav,ind0,jsph) = C_ik(l0,jsph)*(f1*sij*basloc(ind0) + &
                                             & (f2/rijn)*dbasloc(:,ind0))
              elseif ((ksph .eq. jsph) .and. (isph .ne. jsph)) then
                coefY_der(:,icav,ind0,jsph) = -C_ik(l0,jsph)*(f1*sij*basloc(ind0)+ &
                                             & (f2/rijn)*dbasloc(:,ind0))
              else
                coefY_der(:,icav,ind0,jsph) = zero
              endif
            end do ! End of loop m0
          end do ! End of l0
        end do ! End of loop jsph
      end if
      sum_Sjin(igrid,isph) = -(epsp/ddx_data % eps)*sum_int
    end do ! End of loop igrid
  end do ! End of loop isph

  ! Computation of derivative of U_i^e(x_in)
  call fdoga(ddx_data, ksph, sol_sgrid, sum_Sjin, force)

  ! Here, summation over j takes place
  icav = zero
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if(ddx_data % ui(igrid, isph) .gt. zero) then
        ! Extrenal grid point
        icav = icav + 1
        val_dim3(:) = zero
        do jsph = 1, ddx_data % nsph
          do ind0 = 1, nbasis0
            val_dim3(:) = val_dim3(:) + c0_d(ind0,jsph)*coefY_der(:, icav, ind0, jsph)
          end do
        end do
        diff_ep_dim3(:, icav) = val_dim3(:)
      end if
    end do
  end do

  sum_dim3 = zero
  icav = zero
  do isph = 1, ddx_data % nsph
    do igrid =1, ddx_data % ngrid
      if(ddx_data % ui(igrid, isph) .gt. zero) then
        icav = icav + 1
        do ind = 1, ddx_data % nbasis
          sum_dim3(:,ind,isph) = sum_dim3(:,ind,isph) + &
                                & -(epsp/ddx_data % eps)* &
                                & coefvec(igrid, ind, isph)*diff_ep_dim3(:,icav)
        end do
      end if
    end do
  end do

  ! Computation of derivative of \bf(k)_j^l0(x_in)\times Y^j_l0m0(x_in)
  do isph = 1, ddx_data % nsph
    do ind = 1, ddx_data % nbasis
      force = force + sum_dim3(:, ind, isph)*sol_adj(ind, isph)
    end do
  end do
  end subroutine fdouky_f0

  
  !
  ! Subroutine to calculate the third derivative term in C1_C2 matrix, namely the derivative of PU_i
  ! @param[in]  ddx_data     : Input data file
  ! @param[in]  ksph         : Derivative with respect to x_k
  ! @param[in]  Xr           : Solution of the Laplace problem
  ! @param[in]  Xe           : Solution of the HSP problem
  ! @param[in]  Xadj_r_sgrid : Adjoint Laplace solution evaluated at grid point
  ! @param[in]  Xadj_e_sgrid : Adjoint HSP solution evaluated at grid point
  ! @param[in]  diff_re      : l'/r_j[Xr]_jl'm' -(i'_l'(r_j)/i_l'(r_j))[Xe]_jl'm'
  ! @param[out] force        : Force
  subroutine derivative_P(ddx_data, ksph, &
                          & Xr, Xe, &
                          & Xadj_r_sgrid, Xadj_e_sgrid, &
                          & diff_re, force)
  implicit none
  type(ddx_type) :: ddx_data
  integer, intent(in) :: ksph
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph), intent(in) :: Xr, Xe, diff_re
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph), intent(in) :: Xadj_r_sgrid, Xadj_e_sgrid
  real(dp), dimension(3), intent(inout) :: force
  ! Local variable
  ! isph  : Index for sphere i
  ! jsph  : Index for sphere j
  ! igrid : Index for grid point n
  ! l     : l=0,...,lmax
  ! m     : m=-l,...,l
  ! ind   : l^2+l+m+1
  ! l0    : l0=0,...lmax0
  ! m0    : m0=-l0,...,l0
  ! ind0  : l0^2+l0+m0+1
  ! igrid0: Index for grid point n0
  integer :: isph, jsph, igrid, l, m, ind, l0, m0, ind0, igrid0, kep
  ! term  : SK_rijn/SK_rj
  ! termi : DI_ri/SI_ri
  ! termk : DK_ri/SK_ri
  ! sum_int : Intermediate sum
  ! sum_r   : Intermediate sum for Laplace
  ! sum_e   : Intermediate sum for HSP
  real(dp) :: term, termi, termk, sum_int, sum_r, sum_e
  ! rijn  : r_j*r_1^j(x_j^n) = |x_i^n-x_j|
  real(dp) :: rijn
  ! vij   : x_i^n-x_j
  ! sij   : e^j(x_i^n)
  real(dp)  :: vij(3), sij(3)
  ! phi_n_r : Phi corresponding to Laplace problem
  ! phi_n_e : Phi corresponding to HSP problem
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: phi_n_r, phi_n_e
  ! coefY_d : sum_{l0m0} C_ik*term*Y_l0m0^j(x_in)*Y_l0m0(s_n)
  real(dp), dimension(ddx_data % ncav, ddx_data % ngrid, ddx_data % nsph) :: coefY_d
  ! diff_re_sgrid : diff_re evaluated at grid point
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: diff_re_sgrid
  ! basloc : Y_lm(s_n)
  ! vplm   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % nbasis):: basloc, vplm
  ! dbasloc : Derivative of Y_lm(s_n)
  real(dp),  dimension(3, ddx_data % nbasis):: dbasloc
  ! vcos   : Argument to call ylmbas
  ! vsin   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % lmax+1):: vcos, vsin
  ! SK_rijn : Besssel function of first kind for rijn
  ! DK_rijn : Derivative of Besssel function of first kind for rijn
  real(dp), dimension(0:ddx_data % lmax) :: SK_rijn, DK_rijn

  ! Intial allocation of vectors
  sum_int = zero
  sum_r = zero
  sum_e = zero
  phi_n_r = zero
  phi_n_e = zero
  coefY_d = zero
  diff_re_sgrid = zero
  basloc = zero
  vplm = zero
  dbasloc = zero
  vcos = zero
  vsin = zero
  SK_rijn = zero
  DK_rijn = zero

  ! Compute  summation over l0, m0
  ! Loop over the sphers j
  do jsph = 1, ddx_data % nsph
    ! Loop over the grid points n0
    do igrid0 = 1, ddx_data % ngrid
      kep = zero
      ! Loop over spheres i
      do isph = 1, ddx_data % nsph
        ! Loop over grid points n
        do igrid = 1, ddx_data % ngrid
          ! Check for U_i^{eta}(x_in)
          if(ddx_data % ui(igrid, isph) .gt. zero) then
            kep = kep + 1
            vij  = ddx_data % csph(:,isph) + &
                   & ddx_data % rsph(isph)*ddx_data % cgrid(:,igrid) - &
                   & ddx_data % csph(:,jsph)
            rijn = sqrt(dot_product(vij,vij))
            sij = vij/rijn

            call modified_spherical_bessel_second_kind(lmax0, rijn*ddx_data % kappa,&
                                                     & SK_rijn, DK_rijn)
            call dbasis(ddx_data, sij, basloc, dbasloc, vplm, vcos, vsin)
            sum_int = zero
            ! Loop over l0
            do l0 = 0, lmax0
              term = SK_rijn(l0)/SK_ri(l0,jsph)
              ! Loop over m0
              do m0 = -l0,l0
                ind0 = l0**2 + l0 + m0 + 1
                sum_int = sum_int + C_ik(l0, jsph) *term*basloc(ind0)&
                           & *ddx_data % vgrid(ind0,igrid0)
              end do ! End of loop m0
            end do! End of loop l0
            coefY_d(kep, igrid0, jsph) = sum_int
          end if
        end do ! End of loop igrid
      end do! End of loop isph
    end do ! End of loop igrid0
  end do ! End of loop jsph

  ! Compute phi_in
  ! Loop over spheres j
  do jsph = 1, ddx_data % nsph
    ! Loop over grid points n0
    do igrid0 = 1, ddx_data % ngrid
      kep = zero
      sum_r = zero
      sum_e = zero
      ! Loop over sphers i
      do isph = 1, ddx_data % nsph
        ! Loop over grid points n
        do igrid = 1, ddx_data % ngrid
          if(ddx_data % ui(igrid, isph) .gt. zero) then
            kep = kep + 1
            sum_r = sum_r + coefY_d(kep, igrid0, jsph)*Xadj_r_sgrid(igrid, isph) &
                    & * ddx_data % wgrid(igrid)*ddx_data % ui(igrid, isph)
            sum_e = sum_e + coefY_d(kep, igrid0, jsph)*Xadj_e_sgrid(igrid, isph) &
                    & * ddx_data % wgrid(igrid)*ddx_data % ui(igrid, isph)
          end if
        end do
      end do
      phi_n_r(igrid0, jsph) = sum_r
      phi_n_e(igrid0, jsph) = sum_e
    end do! End of loop j
  end do ! End of loop igrid0


  call dgemm('T', 'N', ddx_data % ngrid, ddx_data % nsph, &
            & ddx_data % nbasis, one, ddx_data % vgrid, ddx_data % vgrid_nbasis, &
            & diff_re , ddx_data % nbasis, zero, diff_re_sgrid, &
            & ddx_data % ngrid)
  call fdoga(ddx_data, ksph, diff_re_sgrid, phi_n_r, force)
  call fdoga(ddx_data, ksph, diff_re_sgrid, phi_n_e, force)
  end subroutine derivative_P

  !
  ! Subroutine to calculate the derivative of C_{0l0m0}^j
  ! @param[in]  ddx_data           : Input data file
  ! @param[in]  ksph               : Derivative with respect to x_k
  ! @param[in]  sol_sgrid          : Solution of the Adjoint problem evaluated at the grid
  ! @param[in]  gradpsi            : Gradient of Psi_0
  ! @param[in]  normal_hessian_cav : Normal of the Hessian evaluated at cavity points
  ! @param[in]  icav_g             : Index of outside cavity point
  ! @param[out] force              : Force corresponding to HSP problem
  subroutine fdoco(ddx_data, ksph, sol_sgrid, gradpsi, normal_hessian_cav, icav_g, force)
  implicit none
  type(ddx_type) :: ddx_data
  integer, intent(in) :: ksph
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph), intent(in) :: sol_sgrid
  real(dp), dimension(3, ddx_data % ncav), intent(in) :: gradpsi
  real(dp), dimension(3, ddx_data % ncav), intent(in) :: normal_hessian_cav
  integer, intent(inout) :: icav_g
  real(dp), dimension(3), intent(inout) :: force
  ! Local variable
  ! isph  : Index for sphere i
  ! jsph  : Index for sphere j
  ! igrid : Index for grid point n
  ! l     : l=0,...,lmax
  ! m     : m=-l,...,l
  ! ind   : l^2+l+m+1
  ! l0    : l0=0,...lmax0
  ! m0    : m0=-l0,...,l0
  ! ind0  : l0^2+l0+m0+1
  ! igrid0: Index for grid point n0
  ! kep   : Index for cavity points
  integer :: isph, jsph, igrid, l, m, ind, l0, m0, ind0, igrid0, kep
  ! term  : SK_rijn/SK_rj
  ! termi : DI_ri/SI_ri
  ! termk : DK_ri/SK_ri
  ! sum_int : Intermediate sum
  ! hessian_contribution :
  real(dp) :: term, termi, termk, sum_int, hessian_contribution(3), nderpsi
  ! rijn  : r_j*r_1^j(x_j^n) = |x_i^n-x_j|
  real(dp) :: rijn
  ! vij   : x_i^n-x_j
  ! sij   : e^j(x_i^n)
  real(dp)  :: vij(3), sij(3)
  ! phi_n : Phi corresponding to Laplace problem
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: phi_n
  ! coefY_d : sum_{l0m0} C_ik*term*Y_l0m0^j(x_in)*Y_l0m0(s_n)
  real(dp), dimension(ddx_data % ncav, ddx_data % ngrid, ddx_data % nsph) :: coefY_d
  ! gradpsi_grid : gradpsi evaluated at grid point
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph) :: gradpsi_grid
  ! basloc : Y_lm(s_n)
  ! vplm   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % nbasis):: basloc, vplm
  ! dbasloc : Derivative of Y_lm(s_n)
  real(dp),  dimension(3, ddx_data % nbasis):: dbasloc
  ! vcos   : Argument to call ylmbas
  ! vsin   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % lmax+1):: vcos, vsin
  ! SK_rijn : Besssel function of first kind for rijn
  ! DK_rijn : Derivative of Besssel function of first kind for rijn
  real(dp), dimension(0:ddx_data % lmax) :: SK_rijn, DK_rijn


  ! Intial allocation of vectors
  sum_int = zero
  phi_n = zero
  coefY_d = zero
  gradpsi_grid = zero
  basloc = zero
  vplm = zero
  dbasloc = zero
  vcos = zero
  vsin = zero
  SK_rijn = zero
  DK_rijn = zero


  ! Compute  summation over l0, m0
  ! Loop over the sphers j
  do jsph = 1, ddx_data % nsph
    ! Loop over the grid points n0
    do igrid0 = 1, ddx_data % ngrid
      kep = zero
      ! Loop over spheres i
      do isph = 1, ddx_data % nsph
        ! Loop over grid points n
        do igrid = 1, ddx_data % ngrid
          ! Check for U_i^{eta}(x_in)
          if(ddx_data % ui(igrid, isph) .gt. zero) then
            kep = kep + 1
            vij  = ddx_data % csph(:,isph) + &
                   & ddx_data % rsph(isph)*ddx_data % cgrid(:,igrid) - &
                   & ddx_data % csph(:,jsph)
            rijn = sqrt(dot_product(vij,vij))
            sij = vij/rijn

            call modified_spherical_bessel_second_kind(lmax0, rijn*ddx_data % kappa,&
                                                     & SK_rijn, DK_rijn)
            call dbasis(ddx_data, sij, basloc, dbasloc, vplm, vcos, vsin)
            sum_int = zero
            ! Loop over l0
            do l0 = 0, lmax0
              term = SK_rijn(l0)/SK_ri(l0,jsph)
              ! Loop over m0
              do m0 = -l0,l0
                ind0 = l0**2 + l0 + m0 + 1
                sum_int = sum_int + C_ik(l0, jsph) &
                           & *term*basloc(ind0) &
                           & *ddx_data % vgrid(ind0,igrid0)
              end do ! End of loop m0
            end do! End of loop l0
            coefY_d(kep, igrid0, jsph) = sum_int
          end if
        end do ! End of loop igrid
      end do! End of loop isph
    end do ! End of loop igrid0
  end do ! End of loop jsph

  ! Compute phi_in
  ! Loop over spheres j
  do jsph = 1, ddx_data % nsph
    ! Loop over grid points n0
    do igrid0 = 1, ddx_data % ngrid
      kep = zero
      sum_int = zero
      ! Loop over sphers i
      do isph = 1, ddx_data % nsph
        ! Loop over grid points n
        do igrid = 1, ddx_data % ngrid
          if(ddx_data % ui(igrid, isph) .gt. zero) then
            kep = kep + 1
            sum_int = sum_int + coefY_d(kep, igrid0, jsph)*sol_sgrid(igrid, isph) &
                    & * ddx_data % wgrid(igrid)*ddx_data % ui(igrid, isph)
          end if
        end do
      end do
      phi_n(igrid0, jsph) = -(epsp/ddx_data % eps)*sum_int
    end do! End of loop j
  end do ! End of loop igrid

  kep = zero
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if(ddx_data % ui(igrid, isph) .gt. zero) then
        kep = kep + 1
        nderpsi = dot_product( gradpsi(:,kep),ddx_data % cgrid(:,igrid) )
        gradpsi_grid(igrid, isph) = nderpsi
      end if
    end do ! End of loop igrid
  end do ! End of loop i

  call fdoga(ddx_data, ksph, gradpsi_grid, phi_n, force)

  ! Compute the Hessian contributions
  do igrid = 1, ddx_data % ngrid
    if(ddx_data % ui(igrid, ksph) .gt. zero) then
      icav_g = icav_g + 1
      force = force + ddx_data % wgrid(igrid)*ddx_data % ui(igrid, ksph)*&
                       & phi_n(igrid, ksph)*normal_hessian_cav(:, icav_g)
    end if
  end do

  call fdops(ddx_data, ksph, phi_n, force)

  end subroutine fdoco

  !
  ! fdops : Force derivative of potential at spheres
  ! @param[in]  ddx_data : Input data file
  ! @param[in]  phi_n    : phi_n^j
  ! @param[in]  ksph     : Derivative with respect to x_k
  ! @param[out] force    : Force
  subroutine fdops(ddx_data, ksph, phi_n, force)
  type(ddx_type), intent(in) :: ddx_data
  integer, intent(in) :: ksph
  real(dp),  dimension(ddx_data % ngrid, ddx_data % nsph), intent(in)    :: phi_n
  real(dp),  dimension(3), intent(inout) :: force
  !
  ! Local variables
  ! jsph  : Index for sphere j
  ! i, j  : Index for dimension
  ! igrid : Index for grid points
  integer :: jsph, i,j, igrid
  ! sum_int            : Intermediate sum
  ! vij                : x_i^n-x_j
  ! rijn               : |x_i^n-x_j|
  ! normal_hessian_cav : Normal derivative of Hessian of psi
  ! vij_vijT           : vij*vij^T
  real(dp)  :: sum_int(3), vij(3), rijn, normal_hessian_cav(3), vij_vijT(3,3)
  ! identity_matrix : Identity matrix of size 3x3
  ! hessianv        : Hessian of psi evaluated by centers
  real(dp) :: identity_matrix(3,3), hessianv(3,3)
  real(dp), external :: dnrm2
  ! Create Identity matrix
  identity_matrix = zero
  do i = 1, 3
    identity_matrix(i,i) = one
  end do

  sum_int = zero
  ! Loop over spheres
  do jsph = 1, ddx_data % nsph
    ! Loop over grid points
    do igrid = 1, ddx_data % ngrid
      if(ddx_data % ui(igrid, jsph) .ne. zero) then
        vij_vijT = zero
        hessianv = zero
        normal_hessian_cav = zero
        vij = zero
        vij = ddx_data % csph(:, jsph) + &
           & ddx_data % rsph(jsph)*ddx_data % cgrid(:, igrid) - &
           & ddx_data % csph(:, ksph)
        do i = 1,3
          do j = 1,3
            vij_vijT(i,j) = vij(i)*vij(j)
          end do
        end do
        rijn = dnrm2(3, vij, 1)
        hessianv = 3*vij_vijT/(rijn**5)- &
                 & identity_matrix/(rijn**3)
        do i = 1, 3
          normal_hessian_cav = normal_hessian_cav + &
                             & hessianv(:,i)*ddx_data % cgrid(i,igrid)
        end do
        sum_int = sum_int + &
                 & ddx_data % ui(igrid, jsph)* &
                 & ddx_data % wgrid(igrid)* &
                 & phi_n(igrid, jsph)* &
                 & normal_hessian_cav
      end if
    end do
  end do
  force = force - ddx_data % charge(ksph)*sum_int
  end subroutine fdops

  !
  ! Debug derivative of C1 and C2 in code
  ! @param[in, out] rhs_cosmo : C_1*X_r + C_2*X_e
  ! @param[in, out] rhs_hsp   : C_1*X_r + C_2*X_e
  ! @param[in] Xr             : X_r
  ! @param[in] Xe             : X_e
  !
  subroutine C1_C2(ddx_data, rhs_cosmo, rhs_hsp, Xr, Xe)
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(inout) :: rhs_cosmo, rhs_hsp
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph) :: rhs_plus
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph), intent(in) :: Xr, Xe
  
  integer :: isph, jsph, ig, kep, ind, l1,m1, ind1, ind0, istatus
  real(dp), dimension(3) :: vij, sij
  real(dp), dimension(ddx_data % nbasis,ddx_data % nsph) :: diff_re
  real(dp), dimension(nbasis0,ddx_data % nsph) :: diff0
  real(dp), dimension(ddx_data % ncav) :: diff_ep
  real(dp) :: val

  ! Debug variables
  real(dp) :: termi, termk, term, rijn
  integer :: l, m, igrid, l0, m0
  ! basloc : Y_lm(s_n)
  ! vplm   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % nbasis):: basloc, vplm
  ! dbasloc : Derivative of Y_lm(s_n)
  real(dp),  dimension(3, ddx_data % nbasis):: dbasloc
  ! vcos   : Argument to call ylmbas
  ! vsin   : Argument to call ylmbas
  real(dp),  dimension(ddx_data % lmax+1):: vcos, vsin
  ! SK_rijn : Besssel function of first kind for rijn
  ! DK_rijn : Derivative of Besssel function of first kind for rijn
  real(dp), dimension(0:ddx_data % lmax) :: SK_rijn, DK_rijn
  SK_rijn  = 0
  DK_rijn  = 0

  ! diff_re = epsp/eps*l/ri*Xr - i'(ri)/i(ri)*Xe,
  diff_re = zero 
  do jsph = 1, ddx_data % nsph
    do l = 0, ddx_data % lmax
      do m = -l,l
        ind = l**2 + l + m + 1
        diff_re(ind,jsph) = (epsp/ddx_data % eps)*(l/ddx_data % rsph(jsph)) * &
                            & Xr(ind,jsph) - termimat(l,jsph)*Xe(ind,jsph)
      end do
    end do
  end do

  ! diff0 = Pchi * diff_er, linear scaling
  ! Summation over l',m'
  diff0 = zero 
  do jsph = 1, ddx_data % nsph
    do ind0 = 1, nbasis0
      val = zero
      do ind = 1, ddx_data % nbasis
        val = val + diff_re(ind,jsph)*Pchi(ind,ind0, jsph)
      end do
      diff0(ind0, jsph) = val
    end do
  end do

  ! diff_ep = diff0 * coefY,    COST: M^2*nbasis*Nleb
  ! Summation over l0,m0 and j
  diff_ep = zero
  do kep = 1, ddx_data % ncav
    val = zero
    do jsph = 1, ddx_data % nsph
      do ind0 = 1, nbasis0
        val = val + diff0(ind0,jsph)*coefY(kep,ind0,jsph)
      end do
    end do
    diff_ep(kep) = val 
  end do

  ! Summation over n
  rhs_plus = zero
  kep = 0
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if (ddx_data % ui(igrid,isph).gt.zero) then
        kep = kep + 1
        do ind = 1, ddx_data % nbasis
          rhs_plus(ind,isph) = rhs_plus(ind,isph) + &
                              & coefvec(igrid,ind,isph) &!* diff_re(ind, isph)
                              & *diff_ep(kep)
        end do
      end if
    end do
  end do

  rhs_cosmo = rhs_plus
  rhs_hsp =  rhs_plus
  return
  end subroutine C1_C2  

  subroutine wghpot_debug(ddx_data, gradphi, f )
  use bessel
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  real(dp), dimension(3, ddx_data % ncav),       intent(in)  :: gradphi
  real(dp), dimension(ddx_data % ngrid, ddx_data % nsph),    intent(out) :: f

  integer :: isph, igrid, icav, ind, ind0, jg, l, m, jsph
  real(dp) :: nderphi, sumSijn, rijn, coef_Ylm, sumSijn_pre, termi, &
      & termk, term
  real(dp), dimension(3) :: sijn, vij
  real(dp), allocatable :: SK_rijn(:), DK_rijn(:)

  integer :: l0, m0, NM, kep, istatus
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph) :: c0_d
  real(dp), allocatable :: vplm(:), basloc(:), vcos(:), vsin(:)
  real(dp),  dimension(3, ddx_data % nbasis):: dbasloc

  ! initialize
  allocate(vplm(ddx_data % nbasis),&
           & basloc(ddx_data % nbasis),&
           & vcos(ddx_data % lmax+1),&
           & vsin(ddx_data % lmax+1))
  allocate(SK_rijn(0:lmax0),DK_rijn(0:lmax0))
  icav = 0 ; f(:,:)=0.d0
  c0_d = zero
  SK_rijn = zero
  DK_rijn = zero
  !
  ! Compute c0 Eq.(98) QSM19.SISC
  !
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if ( ddx_data % ui(igrid,isph).ne.zero ) then
        icav = icav + 1
        nderphi = dot_product( gradphi(:,icav),ddx_data % cgrid(:,igrid) )
        c0_d(:, isph) = c0_d(:,isph) + &
                     & ddx_data % wgrid(igrid)*&
                     & ddx_data % ui(igrid,isph)*&
                     & nderphi* &
                     & ddx_data % vgrid(:,igrid)
      end if
    end do
  end do

  ! Computation of F0 using above terms
  ! kep: External grid poitns
  kep = 0
  do isph = 1, ddx_data % nsph
    do igrid = 1, ddx_data % ngrid
      if (ddx_data % ui(igrid,isph).gt.zero) then
        kep = kep + 1
        sumSijn = zero
        ! Loop to compute Sijn
        do jsph = 1, ddx_data % nsph
          sumSijn_pre = sumSijn
          vij  = ddx_data % csph(:,isph) + ddx_data % rsph(isph)&
                & *ddx_data % cgrid(:,igrid) - ddx_data % csph(:,jsph)
          rijn = sqrt(dot_product(vij,vij))
          sijn = vij/rijn

          ! Compute Bessel function of 2nd kind for the coordinates
          ! (s_ijn, r_ijn) and compute the basis function for s_ijn
          call modified_spherical_bessel_second_kind(lmax0, &
                                                    & rijn*ddx_data % kappa, &
                                                    & SK_rijn, DK_rijn)
          !call SPHK_bessel(lmax0,rijn*ddx_data % kappa,NM,SK_rijn,DK_rijn)
          call dbasis(ddx_data, sijn, basloc, dbasloc, vplm, vcos, vsin)

          do l0 = 0,lmax0
            term = SK_rijn(l0)/SK_ri(l0,jsph)
            ! coef_Ylm : (der_i_l0/i_l0 - der_k_l0/k_l0)^(-1)*k_l0(r_ijn)/k_l0(r_i)
            do m0 = -l0, l0
              ind0 = l0**2 + l0 + m0 + 1
              sumSijn = sumSijn + c0_d(ind0,jsph)*C_ik(l0,jsph)*term*basloc(ind0)
            end do
          end do
        end do
        !
        ! Here Intermediate value of F_0 is computed Eq. (99)
        ! Mutilplication with Y_lm and weights will happen afterwards
        f(igrid,isph) = -(epsp/ddx_data % eps)&
                      & *ddx_data % ui(igrid,isph)&
                      & * sumSijn
      end if
    end do
  end do

  deallocate( vplm, basloc, vcos, vsin, SK_rijn, DK_rijn  )
  return
  end subroutine wghpot_debug

  !
  ! Subroutine to compute the Modified Spherical Bessel function of the first kind
  ! @param[in]  lmax     : Data type
  ! @param[in]  argument : Argument of Bessel function
  ! @param[out] SI       : Modified Bessel function of the first kind
  ! @param[out] DI       : Derivative of modified Bessel function of the first kind
  subroutine modified_spherical_bessel_first_kind(lmax, argument, SI, DI)
  use Complex_Bessel
  implicit none
  integer, intent(in) :: lmax
  real(dp), intent(in) :: argument
  real(dp), dimension(0:lmax), intent(out) :: SI, DI
  ! Local Variables
  ! Complex_SI     : Modified Bessel functions of the first kind
  ! argument       : Argument for Complex_SI
  ! scaling_factor : sqrt(pi/2x)
  ! fnu            : Starting argument for I_J(x)
  ! NZ             : Number of components set to zero due to underflow
  ! ierr           : Erro indicator for I_J(x)
  ! l              : l=0,...,lmax
  complex(dp), dimension(0:lmax) :: Complex_SI
  complex(dp) :: complex_argument
  real(dp) :: scaling_factor, fnu
  integer :: NZ, ierr, l

  ! Compute I_(0.5+J) for J:1,...,lmax
  ! NOTE: cbesi computes I_(FNU+J-1)
  fnu = 1.5
  scaling_factor = sqrt(PI/(2*argument))
  ! NOTE: Complex argument is required to call I_J(x)
  complex_argument = argument

  ! Compute for l = 0
  call cbesi(complex_argument, fnu - 1.0, 1, 1, Complex_SI(0), NZ, ierr)
  if (ierr .ne. 0) stop 'Error in computing Bessel function of first kind'
  ! Compute for l = 1,...,lmax
  call cbesi(complex_argument, fnu, 1, lmax, Complex_SI(1:lmax), NZ, ierr)
  if (ierr .ne. 0) stop 'Error in computing Bessel function of first kind'

  ! Store the real part of the complex Bessel functions
  SI = real(Complex_SI)
  ! Converting Modified Bessel to Spherical Modified Bessel
  do l = 0, lmax
    SI(l) = scaling_factor*SI(l)
  end do

  ! Computation of Derivatives of SI
  DI(0) = SI(1)
  do l = 1,lmax
    DI(l)= SI(l-1)-((l+1.0D0)*SI(l))/argument
  end do

  end subroutine modified_spherical_bessel_first_kind

  !
  ! Subroutine to compute the Modified Spherical Bessel function of the second kind
  ! @param[in]  lmax     : Size
  ! @param[in]  argument : Argument of Bessel function
  ! @param[out] SK       : Modified Bessel function of the second kind
  ! @param[out] DK       : Derivative of modified Bessel function of the second kind
  subroutine modified_spherical_bessel_second_kind(lmax, argument, SK, DK)
  use Complex_Bessel
  implicit none
  integer , intent(in) :: lmax
  real(dp), intent(in) :: argument
  real(dp), dimension(0:lmax), intent(out) :: SK, DK
  ! Local Variables
  ! Complex_SK     : Modified Bessel functions of the second kind
  ! argument       : Argument for Complex_SK
  ! scaling_factor : sqrt(pi/2x)
  ! fnu            : Starting argument for K_J(x)
  ! NZ             : Number of components set to zero due to underflow
  ! ierr           : Error indicator for K_J(x)
  ! l              : l=0,...,lmax
  complex(dp), dimension(0:lmax) :: Complex_SK
  complex(dp) :: complex_argument
  real(dp) :: scaling_factor, fnu
  integer :: NZ, ierr, l

  ! Compute K_(0.5+J) for J:1,...,lmax
  ! NOTE: cbesk computes K_(FNU+J-1)
  fnu = 1.5
  scaling_factor = sqrt(2/(PI*argument))
  ! NOTE: Complex argument is required to call K_J(x)
  complex_argument = argument

  ! Compute for l = 0
  call cbesk(complex_argument, fnu - 1.0, 1, 1, Complex_SK(0), NZ, ierr)
  if (ierr .ne. 0) stop 'Error in computing Bessel function of second kind'
  ! Compute for l = 1,...,lmax
  call cbesk(complex_argument, fnu, 1, lmax, Complex_SK(1:lmax), NZ, ierr)
  if (ierr .ne. 0) stop 'Error in computing Bessel function of second kind'

  ! Store the real part of the complex Bessel functions
  SK = real(Complex_SK)
  ! Converting Modified Bessel to Spherical Modified Bessel
  do l = 0, lmax
    SK(l) = scaling_factor*SK(l)
  end do

  ! Computation of Derivatives of SK
  DK(0) = -SK(1)
  do l = 1, lmax
    DK(l) = -SK(l-1) - ((l+1.0D0)*SK(l))/argument
  end do

  end subroutine modified_spherical_bessel_second_kind

 ! Debug routine to check the adjoint system. Compute <y^T,Lx>
  ! @param[in] ddx_data : Data type
  ! @param[in] lx       : Matrix multiplication L*x
  ! @param[in] lstarx   : Matrix multiplication Lstar*x
  subroutine debug_adjoint(ddx_data)
  implicit none
  type(ddx_type), intent(in)  :: ddx_data
  ! Local variable
  ! unit_vector   : Unit vector
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_nbasis_nsph_two
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_nbasis_nsph_one
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: zero_vector_nbasis_nsph
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_cosmo
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_hsp
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_cosmo_adj
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_hsp_adj
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_b_adj
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_b
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_a_adj
  real(dp), dimension(ddx_data % nbasis, ddx_data % nsph):: vector_a
  ! sum_adjoint  : Sum for adjoint system
  ! sum_original : Sum for original system
  integer :: ibasis, isph
  real(dp) :: sum_adjoint_c, sum_original_c
  real(dp) :: sum_adjoint_b, sum_original_b
  real(dp) :: sum_adjoint_a, sum_original_a

  vector_nbasis_nsph_one = two
  vector_nbasis_nsph_two = one

  vector_nbasis_nsph_one(1,1) = three
  vector_nbasis_nsph_two(1,1) = three

  zero_vector_nbasis_nsph = zero
  vector_cosmo = zero; vector_hsp = zero
  vector_cosmo_adj = zero; vector_cosmo_adj = zero
  vector_b = zero; vector_b_adj = zero
  vector_a = zero; vector_a_adj = zero
  first_out_iter = .true.


  call update_rhs(ddx_data, zero_vector_nbasis_nsph, &
            & zero_vector_nbasis_nsph, &
            & vector_cosmo, vector_hsp, &
            & vector_nbasis_nsph_one,&
            & vector_nbasis_nsph_one)
  call update_rhs_adj(ddx_data, zero_vector_nbasis_nsph, &
                & zero_vector_nbasis_nsph, &
                & vector_cosmo_adj, vector_hsp_adj,&
                & vector_nbasis_nsph_two, &
                & vector_nbasis_nsph_two)
  call matABx(ddx_data, ddx_data % n, vector_nbasis_nsph_one, vector_b)
  call bstarx(ddx_data, ddx_data % n, vector_nbasis_nsph_two, vector_b_adj)
  call lx(ddx_data, vector_nbasis_nsph_one, vector_a)
  call lstarx(ddx_data, vector_nbasis_nsph_two, vector_a_adj)

  sum_adjoint_c = 0
  sum_original_c = 0
  sum_adjoint_b = 0
  sum_original_b = 0
  sum_adjoint_a = 0
  sum_original_a = 0


  do ibasis = 1, ddx_data % nbasis
    do isph = 1, ddx_data % nsph
      sum_adjoint_c  = sum_adjoint_c +&
                    & vector_nbasis_nsph_one(ibasis,isph)*&
                    & vector_cosmo_adj(ibasis, isph) + &
                    & vector_nbasis_nsph_one(ibasis,isph)*&
                    & vector_hsp_adj(ibasis, isph)
      sum_original_c = sum_original_c + &
                    & vector_nbasis_nsph_two(ibasis, isph)*&
                    & vector_cosmo(ibasis, isph) + &
                    & vector_nbasis_nsph_two(ibasis,isph)*&
                    & vector_hsp(ibasis, isph)
      sum_adjoint_b = sum_adjoint_b + &
                    & vector_nbasis_nsph_one(ibasis, isph)*&
                    & vector_b_adj(ibasis, isph)
      sum_original_b = sum_original_b + &
                    & vector_nbasis_nsph_two(ibasis, isph)*&
                    & vector_b(ibasis, isph)
      sum_adjoint_a = sum_adjoint_a + &
                    & vector_nbasis_nsph_one(ibasis, isph)*&
                    & vector_a_adj(ibasis, isph)
      sum_original_a = sum_original_a + &
                    & vector_nbasis_nsph_two(ibasis, isph)*&
                    & vector_a(ibasis, isph)
    end do
  end do
  if (ddx_data % iprint .ge. 1) then
    write(*,*) 'The original system <y^T,Cx>  : ', sum_original_c
    write(*,*) 'The adjoint system  <y^T,C*x> : ', sum_adjoint_c
    write(*,*) 'The original system <y^T,Bx>  : ', sum_original_b
    write(*,*) 'The adjoint system  <y^T,B*x> : ', sum_adjoint_b
    write(*,*) 'The original system <y^T,Ax>  : ', sum_original_a
    write(*,*) 'The adjoint system  <y^T,A*x> : ', sum_adjoint_a
  end if
  return
  end subroutine debug_adjoint

  
  subroutine ddlpb_free(ddx_data)
  implicit none
  type(ddx_type), intent(in) :: ddx_data
  integer :: istatus
  if(allocated(SI_ri)) then
    deallocate(SI_ri, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(DI_ri)) then
    deallocate(DI_ri, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(SK_ri)) then
    deallocate(SK_ri, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(DK_ri)) then
    deallocate(DK_ri, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(diff_re_c1)) then
    deallocate(diff_re_c1, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(diff_re_c2)) then
    deallocate(diff_re_c2, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(diff0_c1)) then
    deallocate(diff0_c1, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if


  if(allocated(diff0_c2)) then
    deallocate(diff0_c2, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if


  if(allocated(diff_ep_adj)) then
    deallocate(diff_ep_adj, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if


  if(allocated(diff_ep_c2)) then
    deallocate(diff_ep_c2, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if


  if(allocated(termimat)) then
    deallocate(termimat, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(coefY)) then
    deallocate(coefY, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(C_ik)) then
    deallocate(C_ik, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(coefvec)) then
    deallocate(coefvec, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  if(allocated(Pchi)) then
    deallocate(Pchi, stat=istatus)
    if(istatus .ne. zero) then
      write(*,*) 'ddlpb_free: [1] deallocation failed'
      stop 1
    end if
  end if

  end subroutine ddlpb_free

end module ddx_lpb
