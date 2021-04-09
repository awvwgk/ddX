module ddx_cinterface
    use, intrinsic :: iso_c_binding
    use ddx_core
    use ddx_operators
    use ddx
    implicit none

contains

function ddx_get_supported_lebedev_grids(n, grids) result(c_ngrids) bind(C)
    integer(c_int), intent(in), value ::  n
    integer(c_int), intent(out) :: grids(n)
    integer(c_int) :: c_ngrids
    c_ngrids = min(n, nllg)
    grids(1:c_ngrids) = ng0(1:c_ngrids)
end

function ddx_init(nsph, charge, x, y, z, rvdw, model, lmax, ngrid, force, &
        & fmm, pm, pl, se, eta, eps, kappa, nproc, info) result(c_ddx) bind (C)
    integer(c_int), intent(in), value :: nsph, model, lmax, force, fmm, pm, pl, &
        & ngrid, nproc
    real(c_double), intent(in) :: charge(nsph), x(nsph), y(nsph), z(nsph), &
        & rvdw(nsph)
    real(c_double), intent(in), value :: se, eta, eps, kappa
    integer(c_int), intent(out) :: info
    type(c_ptr) :: c_ddx
    type(ddx_type), pointer :: ddx
    integer :: passproc
    allocate(ddx)
    c_ddx = c_loc(ddx)
    passproc = nproc

    call ddinit(nsph, charge, x, y, z, rvdw, model, lmax, ngrid, force, &
            & fmm, pm, pl, 0, 0, se, eta, eps, kappa, 1, 1d-10, &
            & 100, 20, passproc, ddx, info)
end

subroutine ddx_finish(c_ddx) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    call c_f_pointer(c_ddx, ddx)
    call ddfree(ddx)
    deallocate(ddx)
end

function ddx_get_ncav(c_ddx) result(c_ncav) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    integer(c_int) :: c_ncav
    call c_f_pointer(c_ddx, ddx)
    c_ncav = ddx % ncav
end

function ddx_get_nsph(c_ddx) result(c_nsph) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    integer(c_int) :: c_nsph
    call c_f_pointer(c_ddx, ddx)
    c_nsph = ddx % nsph
end

function ddx_get_nbasis(c_ddx) result(c_nbasis) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    integer(c_int) :: c_nbasis
    call c_f_pointer(c_ddx, ddx)
    c_nbasis = ddx % nbasis
end

function ddx_get_ngrid(c_ddx) result(c_ngrid) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    integer(c_int) :: c_ngrid
    call c_f_pointer(c_ddx, ddx)
    c_ngrid = ddx % ngrid
end

function ddx_get_lmax(c_ddx) result(c_lmax) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    integer(c_int) :: c_lmax
    call c_f_pointer(c_ddx, ddx)
    c_lmax = ddx % lmax
end

function ddx_get_epsilon(c_ddx) result(c_epsilon) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    real(c_double) :: c_epsilon
    call c_f_pointer(c_ddx, ddx)
    c_epsilon = ddx % eps
end

subroutine ddx_get_ccav(c_ddx, ncav, c_ccav) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    integer(c_int), intent(in), value :: ncav
    real(c_double), intent(out) :: c_ccav(3, ncav)
    type(ddx_type), pointer :: ddx
    call c_f_pointer(c_ddx, ddx)
    c_ccav(:, :) = ddx % ccav(:, :)
end


! With reference to a atomic sphere `sphere` of radius `r` centred at `a` compute:
!       4π       x_<^l
!     ------  ----------  Y_l^m(|x - a|)
!     2l + 1   x_>^(l+1)
! where (x_<, x_>) = (|x-a|, r) if x inside the sphere, else (x_<, x_>) = (r, |x-a|)
! lmax should be identical to the value stored inside c_ddx or less.
subroutine ddx_scaled_ylm(c_ddx, lmax, x, sphere, ylm) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    integer(c_int), intent(in), value :: lmax
    real(c_double), intent(in) :: x(3)
    integer(c_int), intent(in), value :: sphere
    real(c_double), intent(out) :: ylm((lmax+1)**2)
    real(dp) :: delta(3)
    real(dp) :: ratio, rho, ctheta, stheta, cphi, sphi, dnorm
    real(dp) :: vplm((lmax+1)**2), vcos(lmax+1), vsin(lmax+1)
    double precision, external :: dnrm2
    integer :: ind, m, l
    type(ddx_type), pointer :: ddx
    call c_f_pointer(c_ddx, ddx)

    delta = x(:) - ddx%csph(:, sphere)
    dnorm = dnrm2(3, delta, 1)
    delta = delta / dnorm
    call ylmbas(delta, rho, ctheta, stheta, cphi, sphi, ddx%lmax, ddx%vscales, &
             &  ylm, vplm, vcos, vsin)
    do l = 0, lmax
        ratio = ddx%v4pi2lp1(l+1) * (dnorm / ddx%rsph(sphere))**l
        ! if (dnorm < ddx%rsph(sphere)) then  ! internal
        !     ratio = ddx%v4pi2lp1(l + 1) * (dnorm / ddx%rsph(sphere))**l
        ! else  ! external
        !     ratio = ddx%v4pi2lp1(l + 1) * (ddx%rsph(sphere) / dnorm)**(l+1)
        ! endif
        do m = -l, l
            ind = l*(l+1) + m + 1
            ylm(ind) = ylm(ind) * ratio
        enddo
    enddo
end


subroutine ddx_mkrhs(c_ddx, nsph, ncav, nbasis, phi_cav, gradphi_cav, psi) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    integer(c_int), intent(in), value :: nsph, ncav, nbasis
    real(c_double), intent(out) :: phi_cav(ncav)
    real(c_double), intent(out) :: gradphi_cav(3, ncav)
    real(c_double), intent(out) :: psi(nbasis, nsph)
    call c_f_pointer(c_ddx, ddx)
    call mkrhs(ddx, phi_cav, gradphi_cav, psi)
end

subroutine ddx_mkxi(c_ddx, nsph, ncav, nbasis, s, xi) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    type(ddx_type), pointer :: ddx
    integer(c_int), intent(in), value :: nsph, ncav, nbasis
    real(c_double), intent(in)  :: s(nbasis, nsph)
    real(c_double), intent(out) :: xi(ncav)
    call c_f_pointer(c_ddx, ddx)
    call ddmkxi(ddx, s, xi)
end

subroutine ddx_solve(c_ddx, nsph, ncav, nbasis, phi_cav, gradphi_cav, psi, &
            & itersolver, tol, maxiter, ndiis, iprint, esolv, xs, s, force) bind(C)
    type(c_ptr), intent(in), value :: c_ddx
    integer(c_int), intent(in), value :: nsph, ncav, nbasis
    integer(c_int), intent(in), value :: itersolver, maxiter, ndiis, iprint
    real(c_double), intent(in), value :: tol
    real(c_double), intent(in)  :: phi_cav(ncav), gradphi_cav(3, ncav)
    real(c_double), intent(in)  :: psi(nbasis, nsph)
    real(c_double), intent(out) :: esolv, xs(nbasis, nsph), s(nbasis, nsph)
    real(c_double), intent(out) :: force(3, nsph)
    type(ddx_type), pointer :: ddx
    integer :: o_itersolver, o_maxiter, o_ndiis, o_iprint
    real(dp) :: o_tol
    call c_f_pointer(c_ddx, ddx)

    ! Save the ddx data and set our solver parameters
    ! (This is temporary until the parameters have moved)
    o_tol        = ddx % tol
    o_itersolver = ddx % itersolver
    o_maxiter    = ddx % maxiter
    o_ndiis      = ddx % ndiis
    o_iprint     = ddx % iprint

    ddx % tol        = tol
    ddx % itersolver = itersolver
    ddx % maxiter    = maxiter
    if (ddx % ndiis .ne. ndiis) then
        write(*, "(A)") "Sorry setting ndiis from ddx_solve currently not supported."
    endif
    ! ddx % ndiis      = min(ndiis, ddx % ndiis)
    ! ddx % iprint     = iprint
    call ddsolve(ddx, phi_cav, gradphi_cav, psi, esolv, force)
    ddx % tol        = o_tol
    ddx % itersolver = o_itersolver
    ddx % maxiter    = o_maxiter
    ddx % ndiis      = o_ndiis
    ddx % iprint     = o_iprint

    ! Extract solution
    xs(:, :) = ddx % xs(:, :)
    s(:, :)  = ddx % s(:, :)
end

end
