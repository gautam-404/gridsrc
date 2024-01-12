! ***********************************************************************
!
!   Copyright (C) 2010  Bill Paxton
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful, 
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************
 
      module run_star_extras

      use star_lib
      use star_def
      use const_def
      use crlibm_lib
      
      implicit none
      
      ! these routines are called by the standard run_star check_model
      contains
      
      subroutine extras_controls(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         ! this is the place to set any procedure pointers you want to change
         ! e.g., other_wind, other_mixing, other_energy  (see star_data.inc)

         ! Uncomment these lines if you wish to use the functions in this file,
         ! otherwise we use a null_ version which does nothing.
         s% extras_startup => extras_startup
         s% extras_check_model => extras_check_model
         s% extras_finish_step => extras_finish_step
         s% extras_after_evolve => extras_after_evolve
         s% how_many_extra_history_columns => how_many_extra_history_columns
         s% data_for_extra_history_columns => data_for_extra_history_columns
         s% how_many_extra_profile_columns => how_many_extra_profile_columns
         s% data_for_extra_profile_columns => data_for_extra_profile_columns  

         ! Once you have set the function pointers you want,
         ! then uncomment this (or set it in your star_job inlist)
         ! to disable the printed warning message,
         s% job% warn_run_star_extras =.false.       

         
      end subroutine extras_controls
 
      ! None of the following functions are called unless you set their
      ! function point in extras_control.
          
      
      integer function extras_startup(id, restart, ierr)
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_startup = 0
         if (.not. restart) then
            call alloc_extra_info(s)
         else ! it is a restart
            call unpack_extra_info(s)
         end if

      end function extras_startup
      

      ! returns either keep_going, retry, backup, or terminate.
      integer function extras_check_model(id, id_extra)
         use crlibm_lib, only: safe_log10_cr
         
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         real(dp) :: min_center_h1_for_diff
         real(dp), parameter :: huge_dt_limit = 3.15d16 ! ~1 Gyr
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_check_model = keep_going         
         if (.false. .and. s% star_mass_h1 < 0.35d0) then
            ! stop when star hydrogen mass drops to specified level
            extras_check_model = terminate
            write(*, *) 'have reached desired hydrogen mass'
            return
         end if

         ! define STOPPING CRITERION: log_Teff_lower_limit = 4.2
         ! if ((s% center_h1 < 1d-4) .and. (safe_log10_cr(s% Teff) < 4.2)) then
         !    termination_code_str(t_xtra1) = 'log_Teff below 4.2'
         !    s% termination_code = t_xtra1
         !    extras_check_model = terminate
         ! end if

         ! check DIFFUSION: to determine whether or not diffusion should happen
         ! no diffusion for fully convective, post-MS, and mega-old models 
         s% diffusion_dt_limit = 3.15d7
         if(abs(s% mass_conv_core - s% star_mass) < 1d-2) then ! => fully convective
            s% diffusion_dt_limit = huge_dt_limit
         end if
         if (s% star_age > 5d10) then !50 Gyr is really old
            s% diffusion_dt_limit = huge_dt_limit
         end if
         min_center_h1_for_diff = 1d-10
         if (s% center_h1 < min_center_h1_for_diff) then
            s% diffusion_dt_limit = huge_dt_limit
         end if

         ! if you want to check multiple conditions, it can be useful
         ! to set a different termination code depending on which
         ! condition was triggered.  MESA provides 9 customizeable
         ! termination codes, named t_xtra1 .. t_xtra9.  You can
         ! customize the messages that will be printed upon exit by
         ! setting the corresponding termination_code_str value.
         ! termination_code_str(t_xtra1) = 'my termination condition'

         ! by default, indicate where (in the code) MESA terminated
         ! if (extras_check_model == terminate) s% termination_code = t_extras_check_model
      end function extras_check_model


      integer function how_many_extra_history_columns(id, id_extra)
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_columns = 5
      end function how_many_extra_history_columns
      
      
      subroutine data_for_extra_history_columns(id, id_extra, n, names, vals, ierr)
         use crlibm_lib, only: safe_log10_cr
         
         integer, intent(in) :: id, id_extra, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         
         integer :: k
         integer :: he_core_zone
         integer :: h_shell_bottom_zone
         integer :: h_shell_top_zone
         integer :: envelope_bottom_zone
         real :: m_env_conv
         real :: m_env_rad
         
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         !note: do NOT add the extras names to history_columns.list
         ! the history_columns.list is only for the built-in log column options.
         ! it must not include the new column names you are adding here.
         
         ! log10(LHe/LH)
         names(1) = "log_LHe_div_LH"
         vals(1) = safe_log10_cr(s% power_he_burn / s% power_h_burn)
            		 	 
   		 ! Determination of he_core_zone
   		 do k = s% nz, 1, -1
   		 	if (s% he_core_mass > 0) then
   		 		if (s% q(k) * s% star_mass >= s% he_core_mass) then
   		 			he_core_zone = k
   		 			exit
   		 		endif
   		 	else
   		 		he_core_zone = -1
   		 	endif
   		 end do
   		 
   		 names(2) = "he_core_zone"
   		 vals(2) = he_core_zone
   		 
   		 names(3) = "he_core_q"
   		 vals(3) = s% he_core_mass / s% star_mass
   		    		 
   		 ! M_env_conv / M_env_rad
   		 ! Note that M_env_conv takes into an account also zones with overshooting
   		 ! and semiconvection.	
         names(4) = "M_env_c_div_M_env_rad"
         m_env_conv = 0.
   		m_env_rad = 0.
         if (s% he_core_mass > 0) then
            do k = he_core_zone, 1, -1
               if (s% mlt_mixing_type(k) >= 1 .and. s% mlt_mixing_type(k) <= 4) then
                  m_env_conv = m_env_conv + s% dm(k)
               else
                  m_env_rad = m_env_rad + s% dm(k)
               endif
            end do         
               vals(4) = m_env_conv / m_env_rad
            else
               vals(4) = -1.
         endif
         
         ! envelope_mass / he_core_mass
         names(5) = "M_he_core_div_M_env"
         vals(5) = s% he_core_mass / (s% star_mass - s% he_core_mass)
         
      end subroutine data_for_extra_history_columns

      
      integer function how_many_extra_profile_columns(id, id_extra)
         use star_def, only: star_info
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_columns = 7
      end function how_many_extra_profile_columns
      
      
      subroutine data_for_extra_profile_columns(id, id_extra, n, nz, names, vals, ierr)
         use star_def, only: star_info, maxlen_profile_column_name
         use const_def, only: dp
         use star_lib, only: star_interp_val_to_pt
         use crlibm_lib, only: safe_log_cr, safe_log10_cr
         integer, intent(in) :: id, id_extra, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         integer :: k
         
         ! Variables needed for osc
         real(dp) :: sm, rt, sl, teff, reff, nc, nt, ng, ar6nc, omega, mean_rho, alpha
         real(dp) :: P, rho, grada, gamma1, chiT, chirho, opacity, d_opacity_dlnT, &
            d_opacity_dlnd, d_eos4_dlnT, logT
         real(dp) :: eps_nuc, d_epsnuc_dlnd, d_epsnuc_dlnT
         real(dp) :: A(nz, 20)
         character fname*50, fname2*540, fname3*50
         integer :: i, ic
         logical :: iscenter
         
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         !note: do NOT add the extra names to profile_columns.list
         ! the profile_columns.list is only for the built-in profile column options.
         ! it must not include the new column names you are adding here.

         ! here is an example for adding a profile column
         !if (n /= 1) stop 'data_for_extra_profile_columns'
         !names(1) = 'beta'
         !do k = 1, nz
         !   vals(k,1) = s% Pgas(k)/s% P(k)
         !end do
         
         
         names(1) = 'rho_face'
         do k = 1, nz
			vals(k,1) = s% rho_face(k)
         end do
         
         names(2) = 'P_face'
         do k = 1, nz
         	vals(k,2) = star_interp_val_to_pt(s% P,k,s% nz,s% dq,'P_face')
         end do
         
         names(3) = 'dlnRho_dlnT_const_Pgas_face'
         do k = 1, nz
         	vals(k,3) = star_interp_val_to_pt(s% dlnRho_dlnT_const_Pgas,k,s% nz,s% dq, &
         	 	'dlnRho_dlnT_const_Pgas_face')
         end do
         
         names(4) = 'opacity_face'
         do k = 1, nz
         	vals(k,4) = star_interp_val_to_pt(s% opacity,k,s% nz,s% dq,'opacity')
         end do
         
         names(5) = 'T_face'
         do k = 1, nz
         	vals(k,5) = star_interp_val_to_pt(s% T,k,s% nz,s% dq,'T_face')
         end do
         
         names(6) = 'chiRho_face'
         do k = 1, nz
         	vals(k,6) = star_interp_val_to_pt(s% chiRho,k,s% nz,s% dq,'chiRho_face')
         end do
         
         names(7) = 'chiT_face'
         do k = 1, nz
         	vals(k,7) = star_interp_val_to_pt(s% chiT,k,s% nz,s% dq,'chiT_face')
         end do
		
         if (s% x_logical_ctrl(1) .and. mod(s% model_number,s% profile_interval) == 0) then
            call make_osc(id, id_extra, n, nz, names, ierr)
         end if
		
      end subroutine data_for_extra_profile_columns
      

      ! returns either keep_going or terminate.
      ! note: cannot request retry or backup; extras_check_model can do that.
      integer function extras_finish_step(id, id_extra)
         use chem_def, only: ih1
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_finish_step = keep_going
         call store_extra_info(s)

         ! to save a profile, 
            ! s% need_to_save_profiles_now = .true.
         ! to update the star log,
            ! s% need_to_update_history_now = .true.
          
         if (abs(s% mstar_dot) > 0. .and. s% xa(s% net_iso(ih1), s% nz) > 0.1) then
            s% varcontrol_target = 5.0d-4
         else if (s% xa(s% net_iso(ih1), s% nz) <= 0.1) then
            s% varcontrol_target = 1.0d-4
         end if

         ! see extras_check_model for information about custom termination codes
         ! by default, indicate where (in the code) MESA terminated
         if (extras_finish_step == terminate) s% termination_code = t_extras_finish_step
      end function extras_finish_step
      
      
      subroutine extras_after_evolve(id, id_extra, ierr)
         integer, intent(in) :: id, id_extra
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return		
      end subroutine extras_after_evolve
      
      
      ! routines for saving and restoring extra data so can do restarts
         
         ! put these defs at the top and delete from the following routines
         !integer, parameter :: extra_info_alloc = 1
         !integer, parameter :: extra_info_get = 2
         !integer, parameter :: extra_info_put = 3
      
      
      subroutine alloc_extra_info(s)
         integer, parameter :: extra_info_alloc = 1
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_alloc)
      end subroutine alloc_extra_info
      
      
      subroutine unpack_extra_info(s)
         integer, parameter :: extra_info_get = 2
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_get)
      end subroutine unpack_extra_info
      
      
      subroutine store_extra_info(s)
         integer, parameter :: extra_info_put = 3
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_put)
      end subroutine store_extra_info
      
      
      subroutine move_extra_info(s,op)
         integer, parameter :: extra_info_alloc = 1
         integer, parameter :: extra_info_get = 2
         integer, parameter :: extra_info_put = 3
         type (star_info), pointer :: s
         integer, intent(in) :: op
         
         integer :: i, j, num_ints, num_dbls, ierr
         
         i = 0
         ! call move_int or move_flg    
         num_ints = i
         
         i = 0
         ! call move_dbl       
         
         num_dbls = i
         
         if (op /= extra_info_alloc) return
         if (num_ints == 0 .and. num_dbls == 0) return
         
         ierr = 0
         call star_alloc_extras(s% id, num_ints, num_dbls, ierr)
         if (ierr /= 0) then
            write(*,*) 'failed in star_alloc_extras'
            write(*,*) 'alloc_extras num_ints', num_ints
            write(*,*) 'alloc_extras num_dbls', num_dbls
            stop 1
         end if
         
         contains
         
         subroutine move_dbl(dbl)
            real(dp) :: dbl
            i = i+1
            select case (op)
            case (extra_info_get)
               dbl = s% extra_work(i)
            case (extra_info_put)
               s% extra_work(i) = dbl
            end select
         end subroutine move_dbl
         
         subroutine move_int(int)
            integer :: int
            i = i+1
            select case (op)
            case (extra_info_get)
               int = s% extra_iwork(i)
            case (extra_info_put)
               s% extra_iwork(i) = int
            end select
         end subroutine move_int
         
         subroutine move_flg(flg)
            logical :: flg
            i = i+1
            select case (op)
            case (extra_info_get)
               flg = (s% extra_iwork(i) /= 0)
            case (extra_info_put)
               if (flg) then
                  s% extra_iwork(i) = 1
               else
                  s% extra_iwork(i) = 0
               end if
            end select
         end subroutine move_flg
      
      end subroutine move_extra_info
      
      subroutine make_osc(id, id_extra, n, nz, names, ierr)
      	! Create osc files. Based on P. Walczak's code.
         use star_def, only: star_info
         use const_def, only: dp
         use crlibm_lib, only: safe_log_cr, safe_log10_cr
         integer, intent(in) :: id, id_extra, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         integer :: k
         
         ! Variables needed for osc
         real(dp) :: sm, rt, sl, teff, reff, nc, nt, ng, ar6nc, omega, mean_rho, alpha
         real(dp) :: P, rho, grada, gamma1, chiT, chirho, opacity, d_opacity_dlnT, &
            d_opacity_dlnd, d_eos4_dlnT, logT
         real(dp) :: eps_nuc, d_epsnuc_dlnd, d_epsnuc_dlnT
         real(dp) :: A(nz, 20)
         character fname*50, fname2*540, fname3*50
         integer :: i, ic
         logical :: iscenter
         
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         fname2=trim(s% model_profile_filename)//".osc"
         fname3=trim(fname2)
         write(fname,123) s% model_number
         123 format("LOGS/osc",i5.5)
         open (234,file=fname,status='unknown')
      
         sm = s% mstar / msol
         rt = 10.0 ** s% log_surface_radius
         sl = 10.0 ** s%log_surface_luminosity
         teff = s% Teff
         reff = dsqrt(sl * lsol / (pi4 * boltz_sigma * s%Teff ** 4.0)) / rsol
         nc = 0
         ng = 1
         ar6nc = 0
         omega = s% omega(1)
         mean_rho = (s% mstar) * 3.0 / (pi4 * ((rt * rsol) ** 3.0))
         
         do i = s% nz, 2, -1
            ic = s% nz - i + 1 
            alpha = s% dq(i - 1) / (s% dq(i - 1) + s% dq(i))
            P = alpha * s% P(i) + (1 - alpha) * s% P(i-1)
            rho = alpha * s%rho(i) + (1 - alpha) * s% rho(i - 1)
            grada = alpha * s% grada(i) + (1 - alpha) * s% grada(i - 1)
            gamma1 = alpha * s% gamma1(i) + (1 - alpha) * s% gamma1(i - 1)
            chiT = alpha * s% chiT(i) + (1 - alpha) * s% chiT(i - 1)
            chirho = alpha * s% chirho(i) + (1 - alpha) * s% chirho(i - 1)
            opacity = alpha * s% opacity(i) + (1 - alpha) * s% opacity(i - 1)
            d_opacity_dlnT = alpha * s% d_opacity_dlnT(i) + (1 - alpha) * s% d_opacity_dlnT(i - 1)
            d_opacity_dlnd = alpha * s% d_opacity_dlnd(i) + (1 - alpha) * s% d_opacity_dlnd(i - 1)
            d_eos4_dlnT = alpha * s% d_eos_dlnT(4,i) + (1 - alpha) * s% d_eos_dlnT(4, i - 1) + &
               alpha * s% d_eos_dlnd(4, i) * s% grad_density(i) / s% gradT(i) + &
               (1 - alpha) * s% d_eos_dlnd(4, i - 1) * s% grad_density(i - 1) / s% gradT(i - 1)
            logT = alpha * safe_log10_cr(s%T(i)) + (1 - alpha) * safe_log10_cr(s% T(i - 1))
            eps_nuc = alpha * s% eps_nuc(i) + (1 - alpha) * s% eps_nuc(i - 1)
            d_epsnuc_dlnT = alpha * s% d_epsnuc_dlnT(i) + (1 - alpha) * s% d_epsnuc_dlnT(i - 1)
            d_epsnuc_dlnd = alpha * s% d_epsnuc_dlnd(i) + (1 - alpha) * s% d_epsnuc_dlnd(i - 1)

            A(i,1) = safe_log_cr(s% r(i) / (rt * rsol))
            A(i,2) = pi4 * s% r(i) ** 3.0 * mean_rho / (s% m(i))
            A(i,3) = s% r(i) * s% grav(i) * rho / P
            A(i,4) = 1.0 / gamma1
            A(i,5) = pi4 * s% r(i) ** 3 * rho / s% m(i)
            A(i,6) = s% brunt_N2(i) * s% r(i) / s% grav(i)
            A(i,7) = -chiT / chirho
            A(i,8) = s% gradT(i)
            A(i,9) = grada * (d_opacity_dlnT / opacity - 4.0) + &
               A(i,4) * d_opacity_dlnd / opacity + grada / A(i,8) - d_eos4_dlnT
            A(i,10) = grada / s% gradT(i)
            A(i,11) = d_opacity_dlnT / opacity + d_opacity_dlnd / opacity * A(i,7) - 4.0
            A(i,12) = sl * lsol / s% L(i)
            A(i,13) = sqrt(pi4 * standard_cgrav * mean_rho) / (sl * lsol) * pi4 * &
               s% r(i) ** 3 * P * (-A(i,7)) / grada
            A(i,14) = logT
            A(i,15) = eps_nuc
         
            if (eps_nuc .ne. 0) then
               A(i,16) = d_epsnuc_dlnT / eps_nuc
               A(i,17) = d_epsnuc_dlnd / eps_nuc
            else if (eps_nuc .eq. 0) then
               A(i,16)=0
               A(i,17)=0
            end if
         
            if ((A(i,10) .gt. 1.0) .and. iscenter) then
               nc = ic
               ar6nc = A(i,6)
               iscenter = .false.
            end if
         
            A(i,18) = s% omega(i)
            A(i,19) = (s% gradT(i) * sl * lsol) / (s% gradr(i) * s% L(i))
         end do
         
   ! 		write(234,*) "M R L Teff Reff nc nz-1 ng ar6nc omega_surf"
         write(234,*) sm, rt, sl, teff, reff, NINT(nc), s% nz-1, NINT(ng), ar6nc, omega
   ! 		write(234,*) "a1 a2 V 1_div_Gamma U A a7 a8 a9 a10 a11 a12 a13 logT eNUC dlnenuc_dlnT dlnenuc_dlnrho omega"
         do i = s% nz, 2, -1
            write(234,124) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5), A(i,6), A(i,7), &
               A(i,8), A(i,9), A(i,10), A(i,11), A(i,12), A(i,13), A(i,14), A(i,15), &
               A(i,16), A(i,17), A(i,18)
         end do
         124 format(18(e20.12,x))
         close(234)  
      end subroutine make_osc

      end module run_star_extras


