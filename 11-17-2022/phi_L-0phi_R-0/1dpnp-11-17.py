#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:55:21 2022

@author: douglas
"""
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import os, subprocess, sys
#from custom_colormaps import create_colormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#from matplotlib.ticker import (MultipleLocator,
  #                              FormatStrFormatter,
 #                              AutoMinorLocator)

#############  Physical Constants ################################
q_ele = 1.602E-19
epo = 8.854E-12
kappa = 78.4 # dielectric constant of water
ep = epo*kappa

# https://www.aqion.de/site/diffusion-coefficients
# https://www.physiologyweb.com/calculators/diffusion_time_calculator.html
Danion = 1.33e-9 #Na+ m^2/s
Dcation = 2.03e-9# Cl- m^2/s

# https://is.muni.cz/el/sci/podzim2016/C4020/um/pom/Ionic_Conductivity_and_Diffusion_at_Infinite_Dilution.pdf
#Lanion = 
Lo_anion = 50.08E-4/6.0221408E23

#Lcation =
Lo_cation = 76.31E-4/6.0221408E23


#Danion = 1.0 #Na+ m^2/s
#Dcation = 1.0 # Cl- m^2/s


###############  Adjustable Parameters
Z_anion = 1.00
Z_cation = -1.0


plt.rcParams["font.family"] = "times"
plt.rcParams['text.usetex'] = True

Lx = 0.1
Nx = 100



cn = 5.0E14
#cn = 0.0

phi_left = 0.0
phi_right = 0.0
temperature = 300 # temperature in Kelvin



########### Descretize Time and Space & Initiallize arrays #################

#===============================================================

## Stability Criterion:  dt <= dx^2/(2*D)
## https://www.atmos.albany.edu/facstaff/brose/classes/ATM623_Spring2015/Notes/Lectures/Lecture16%20--%20Numerical%20methods%20for%20diffusion%20models.html

#================================================================

dx = Lx/Nx
x = np.linspace(0,Lx, Nx+1)
dt_min = (dx**2)/(2*max(Danion,Dcation))

##  Typical Diffusion Time Scales  https://www.physiologyweb.com/calculators/diffusion_time_calculator.html
Trun = 220.0 # length of "experiment" in hours
dt = 3600/20
print("dt=", dt)
if dt > dt_min : sys.exit("dt is too large")   


Nt = int(Trun*3600/dt)
print("Nt =", Nt)
Nplot = 200
Ntplot = Nt/Nplot
print("Ntplot = ", Ntplot)


#dt_min = 2.0*np.trunc(0.5*(dx**2)/(2*max(Danion,Dcation))) # makes dt_min an even number

#dt = dt_min
#dt =  5*np.floor(dt_min/5)


Ttotal = Nt*dt
print(dt_min)
print(dt)

t = np.linspace(0, Ttotal, Nt+1)

cn_anion = np.zeros((Nt+1, Nx+1))
cn_cation = np.zeros((Nt+1, Nx+1))
rho_plot = np.zeros((Nt+1, Nx+1))

matrix = np.zeros((Nx+1, Nx+1))
inverse = np.zeros((Nx+1, Nx+1))

rho = np.zeros((Nx+1))
rho_ep = np.zeros(Nx+1)

phi = np.zeros(( Nt+1,Nx+1))
Ex = np.zeros((Nt+1,Nx+1))
m_anion = np.zeros((Nt+1))
m_cation = np.zeros((Nt+1))


#################### Calculation of Poisson Inverse Matrix ################33
for j in np.arange(0, Nx+1):
    for i in np.arange(0, Nx+1):
        if (i >= j):
            element = -(j+1)*(Nx+1-i)/(Nx+1+1)
        elif (i < j):
            element = -(i+1)*(Nx+1-j)/(Nx+1+1)
        inverse[i,j] = element

########## Initial Ion Distrubtion ######################################
for ix in np.arange(0, Nx+1):

    # cn = constant
    #cn_anion[0,ix] = cn
    #cn_cation[0,ix] = cn
    
    # cn = Inverse parabola/concentrated in center 
    cn_anion[0,ix] = cn*x[ix]*(Lx-x[ix])/Lx**2
    cn_cation[0,ix] = cn*x[ix]*(Lx-x[ix])/Lx**2
    
    #  cn = linear w/ max at electrodes
    #cn_anion[0,ix] = cn*x[ix]/Lx
    #cn_cation[0,ix] = cn*(1-x[ix]/Lx)

    # cn = sheath
    #n=10
    #cn_anion[0,ix] = cn*((x[ix]-Lx)**n)/Lx**n
    #cn_cation[0,ix] = cn*x[ix]**n/Lx**n
    
    m_anion[0] = cn_anion[0,ix]*dx/(cn*Lx) + m_anion[0]
    m_cation[0] = cn_cation[0,ix]*dx/(cn*Lx) + m_cation[0]
    
    
    rho[ix] = (Z_anion*q_ele*cn_anion[0,ix]+Z_cation*q_ele*cn_cation[0,ix])
    if (ix ==0):
        rho_ep[0] = -((Z_anion*q_ele*cn_anion[0,ix]+Z_cation*q_ele*cn_cation[0,ix])/ep)*dx**2 - phi_left
    elif (ix==Nx):
        rho_ep[Nx] = -((Z_anion*q_ele*cn_anion[0,ix]+Z_cation*q_ele*cn_cation[0,ix]))/ep*dx**2 - phi_right  
    else:
        rho_ep[ix] = -rho[ix]/ep*dx**2
       
#################   Calculates  initial potential using rho/ep & inverse poission matrix ####        



phi[0,:] = inverse @ rho_ep

for ix in np.arange(0, Nx+1):
    if (ix == 0):
        Ex[0, 0] = -1.0*(-3*phi[0, 0]+4*phi[0, 1]-phi[0, 2])/(2.0*dx)
    elif (ix == Nx):
        Ex[0, Nx] = -1.0*(3*phi[0, Nx] - 4*phi[0,Nx-1] + phi[0,Nx-2])/(2.0*dx)
    else:
        Ex[0, ix] = -1.0*(phi[0, ix+1]-phi[0, ix-1])/(2.0*dx)


#################### Main Loop  ################################################
for it in range(1, Nt+1) :
    #print("it = ", it, " of ", Nt+1)
    for ix in np.arange(1, Nx) :
        #print(ix)
        if (ix == 1):
            #  Note:  Can use central difference since we have x_0, x_1, and x_2
            #          Since cn(x_0) = cn(x_1) [Boundary condition] => formula below
            #cn_anion[it,ix] = cn_anion[it-1,ix] + (Danion*(cn_anion[it-1,ix+1]-cn_anion[it-1,ix])/dx**2)*dt #+ Lo_anion*cn_anion[it-1,ix]*rho_ep[ix]/dx**2*dt
            #cn_cation[it,ix] = cn_cation[it-1,ix] + (Dcation*(cn_cation[it-1,ix+1]-cn_cation[it-1,ix])/dx**2)*dt #+ Lo_cation*cn_cation[it-1,ix]*rho_ep[ix]/dx**2*dt
            Diffusion_anion =  (Danion*(cn_anion[it-1,ix+1]-cn_anion[it-1,ix])/dx**2)   
            Diffusion_cation =  (Dcation*(cn_cation[it-1,ix+1]-cn_cation[it-1,ix])/dx**2) 
            sigma_dEx_anion = Lo_anion*cn_anion[it-1,ix]*rho[ix]/ep
            sigma_dEx_cation = Lo_cation*cn_cation[it-1,ix]*rho[ix]/ep

            cn_anion[it,ix] = cn_anion[it-1,ix] + Diffusion_anion*dt + sigma_dEx_anion *dt
            cn_cation[it,ix] = cn_cation[it-1,ix] + Diffusion_cation*dt + sigma_dEx_cation *dt
           
            cn_cation[it,0] = cn_cation[it,1]
            cn_anion[it,0] = cn_anion[it,1]
            #print(cn_anion[it,ix], cn_cation[it,ix])
        elif (ix == Nx-1) :
            #  Note:  Can use central difference since we have x_N-2, x_N-1, and x_N
            #          Since cn(x_N-1) = cn(x_N) [Boundary condition] => formula below
            #cn_anion[it,ix] = cn_anion[it-1,ix] + (Danion*(cn_anion[it-1,ix-1]-cn_anion[it-1,ix])/dx**2)*dt# + Lo_anion*cn_anion[it-1,ix]*rho_ep[ix]/dx**2*dt
            #cn_cation[it,ix] = cn_cation[it-1,ix] + (Dcation*(cn_cation[it-1,ix-1]-cn_cation[it-1,ix])/dx**2)*dt# + Lo_anion*cn_anion[it-1,ix]*rho_ep[ix]/dx**2*dt

            Diffusion_anion = (Danion*(cn_anion[it-1,ix-1]-cn_anion[it-1,ix])/dx**2)
            Diffusion_cation = (Dcation*(cn_cation[it-1,ix-1]-cn_cation[it-1,ix])/dx**2)
            sigma_dEx_anion = Lo_anion*cn_anion[it-1,ix]*rho[ix]/ep
            sigma_dEx_cation = Lo_cation*cn_cation[it-1,ix]*rho[ix]/ep

            cn_anion[it,ix] = cn_anion[it-1,ix] + Diffusion_anion*dt + sigma_dEx_anion*dt
            cn_cation[it,ix] = cn_cation[it-1,ix] + Diffusion_cation*dt + sigma_dEx_cation*dt

            cn_anion[it,Nx] = cn_anion[it,Nx-1]
            cn_cation[it,Nx] = cn_cation[it,Nx-1]
            #print(cn_anion[it,ix], cn_cation[it,ix])
        else :
            #cn_anion[it,ix] = cn_anion[it-1,ix] + (Danion*(cn_anion[it-1,ix+1]-2*cn_anion[it-1,ix]+cn_anion[it-1,ix-1])/dx**2)*dt # + Lo_anion*cn_anion[it-1,ix]*rho_ep[ix]/dx**2*dt
            #cn_cation[it,ix] = cn_cation[it-1,ix] + (Dcation*(cn_cation[it-1,ix+1]-2*cn_cation[it-1,ix]+cn_cation[it-1,ix-1])/dx**2)*dt #+ Lo_cation*cn_cation[it-1,ix]*rho_ep[ix]/dx**2*dt
            Diffusion_anion = (Danion*(cn_anion[it-1,ix+1]-2*cn_anion[it-1,ix]+cn_anion[it-1,ix-1])/dx**2)
            Diffusion_cation = (Dcation*(cn_cation[it-1,ix+1]-2*cn_cation[it-1,ix]+cn_cation[it-1,ix-1])/dx**2)
            sigma_dEx_anion = Lo_anion*cn_anion[it-1,ix]*rho[ix]/ep
            sigma_dEx_cation = Lo_cation*cn_cation[it-1,ix]*rho[ix]/ep
        
            
            
            cn_anion[it,ix] = cn_anion[it-1,ix] + Diffusion_anion*dt + sigma_dEx_anion*dt   
            cn_cation[it,ix] = cn_cation[it-1,ix] + Diffusion_cation*dt  + sigma_dEx_anion*dt   

        #print(Diffusion_anion, sigma_dEx_anion)    

        rho[ix] = (Z_anion*q_ele*cn_anion[it,ix]+Z_cation*q_ele*cn_cation[it,ix])
        if (ix ==0):
            rho_ep[0] = -1.0*((Z_anion*q_ele*cn_anion[it,ix]+Z_cation*q_ele*cn_cation[it,ix])/ep)*dx**2 - phi_left
        elif (ix==Nx):
            rho_ep[Nx] = -1.0*((Z_anion*q_ele*cn_anion[it,ix]+Z_cation*q_ele*cn_cation[it,ix]))/ep*dx**2 - phi_right  
        else:
            rho_ep[ix] = -1.0*rho[ix]/ep*dx**2
       
    phi[it,:] = inverse @ rho_ep
    

    for ix in np.arange(0, Nx+1) :
        if (ix ==0):
            Ex[it,0] = -1.0*(-3*phi[it,0]+4*phi[it,1]-phi[it,2])/(2.0*dx)
        elif (ix == Nx):
            Ex[it,Nx]= -1.0*(3*phi[it,Nx] - 4*phi[it,Nx-1] + phi[it,Nx-2])/(2.0*dx)
        else :
            Ex[it,ix] = -1.0*(phi[it,ix+1]-phi[it,ix-1])/(2.0*dx)
            
    for ix in np.arange(0, Nx+1) :
        m_anion[it] = cn_anion[it,ix]*dx/(cn*Lx) + m_anion[it]
        m_cation[it] = cn_cation[it,ix]*dx/(cn*Lx) + m_cation[it]
    


########## Plot of Concentrations ##########################
cnmax = 2.0*np.trunc(0.5*max(np.amax(cn_cation),np.amax(cn_anion)))
it = 0
iplot = 0
iframe = 1  
while (it <= Nt):

    if(it == 0 or iplot == Ntplot):

        cn_plate = plt.figure(it, frameon=True, figsize=[6.0, 6.0],dpi = 300, constrained_layout= True)
        ax = plt.subplot(111)
        ax=plt.gca()   
        ax.set_facecolor('#74ccf4')

  
        plt.rcParams.update({'font.size': 12})
        plt.grid(visible=True, which="minor", axis='y', color='black', linestyle='-', linewidth=2)

        plt.xlabel(r"\textbf{x~[m]}")
        plt.ylabel(r'\textbf{Concentration}  $\mathbf{[m^{-3}]}$')


        plt.xlim(0, Lx)
        plt.ylim(0, cnmax*1.1)


        time = it*dt/3600
        Ttotal_hr = Ttotal/3600
 
        plt.title("Diffusion of NaCl in water \n t = %5.2f hr/ %5.2f hr \n" %(time, Ttotal_hr), fontsize=12)
        
        plt.plot(x[:],cn_cation[it,:], color = "#30583b", label = "Cl")
        plt.plot(x,cn_anion[it,:], color='#ffc65a', label = "Na")

       
        plt.grid(visible=True, which="both", axis='both', color='black', linestyle='--', linewidth=1)         #plt.xlabel("L [m]", fontsize=18)

        plt.ylabel(r'\textbf{Concentration}  $\mathbf{[m^{-3}]}$', fontsize=12)


        #plt.legend(loc="upper left", fontsize=12)
        #plt.legend(bbox_to_anchor =(0.5,-0.5), loc='lower center')
        
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.05),
          fancybox=True, shadow=True, ncol=5)
    

        cn_plate_name = "cn_plate_%04d.png" % (iframe)

        plt.savefig(cn_plate_name, dpi='figure', format='png', bbox_inches=None, pad_inches=0.25, transparent=False)
        iplot = 1
        iframe = iframe + 1

    it = it + 1
    iplot = iplot +1
    plt.ioff()
    plt.close(cn_plate)
    


#print("################################################################")
#cmd = "convert plate_*.png movie.gif"
cmd0 = "rm cn.webm"
os.system(cmd0)
cmd1 = "ffmpeg -hide_banner -loglevel panic -framerate 20 -i cn_plate_%04d.png cn.webm"
os.system(cmd1)
cmd2 = "rm cn*.png"
os.system(cmd2)


############ Plot of Charge Density ###############################

cn_anion_max = np.amax(cn_anion)
cn_cation_max = np.amax(cn_cation)

it = 0
iplot = 0
iframe = 1  
while (it <= Nt):

    if(it == 0 or iplot == Ntplot):

        rho_plate = plt.figure(it, frameon=True, figsize=[6.0, 6.0],dpi = 300, constrained_layout= True)
        ax = plt.subplot(111)
        ax=plt.gca()   
        #ax.set_facecolor('#74ccf4')

  
        plt.rcParams.update({'font.size': 12})
        #plt.grid(visible=True, which="minor", axis='y', color='black', linestyle='-', linewidth=2)

        plt.xlabel(r"\textbf{x~[m]}")
        plt.ylabel(r'$\mathbf{\rho ~ [\frac{C}{m^3}]}$')


        plt.xlim(0, Lx)
        #plt.ylim(Z_cation*q_ele*cn_cation_max*1.1, Z_anion*q_ele*cn_anion_max*1.1)
        plt.ylim(-2.0E-6, 2.0E-6)


        time = it*dt/3600
        Ttotal_hr = Ttotal/3600
 
        plt.title("Diffusion of NaCl in water \n t = %5.2f hr/ %5.2f hr \n" %(time, Ttotal_hr), fontsize=12)
        

        rho_plot[it,:] = (Z_anion*q_ele*cn_anion[it,:] + Z_cation*q_ele*cn_cation[it,:])
        
        
        plt.plot(x[:],rho_plot[it,:], color = "#ffc65a")
        #plt.plot(x,cn_anion[it,:], color='#ffc65a', label = "Na")

       
        plt.grid(visible=True, which="both", axis='both', color='black', linestyle='--', linewidth=1)         #plt.xlabel("L [m]", fontsize=18)

        #plt.ylabel(r'\textbf{Concentration}  $\mathbf{[m^{-3}]}$', fontsize=12)


        #plt.legend(loc="upper left", fontsize=12)
        #plt.legend(bbox_to_anchor =(0.5,-0.5), loc='lower center')
        
        # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #          box.width, box.height * 0.9])

        # # Put a legend below current axis
        # ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.05),
        #   fancybox=True, shadow=True, ncol=5)
    

        rho_plate_name = "rho_plate_%04d.png" % (iframe)

        plt.savefig(rho_plate_name, dpi='figure', format='png', bbox_inches=None, pad_inches=0.25, transparent=False)
        iplot = 1
        iframe = iframe + 1

    it = it + 1
    iplot = iplot +1
    plt.ioff()
    plt.close(rho_plate)
    


#print("################################################################")
#cmd = "convert plate_*.png movie.gif"
cmd0 = "rm rho.webm"
os.system(cmd0)
cmd1 = "ffmpeg -hide_banner -loglevel panic -framerate 20 -i rho_plate_%04d.png rho.webm"
os.system(cmd1)
cmd2 = "rm rho*.png"
os.system(cmd2)





# for i in np.arange(1, Nx):
#     phi_plot[i] = phi[i-1]
    

########### Analytic Solution of Potential Computation for Check on Numerics ##############

phi_analytic = np.zeros((Nx+1))

for i in np.arange(0, Nx+1):
    #x[i] = i*dx
    ########### Solution for cn = constant, phi_left = 00., phi_right = 0.0
    # C_analytic = phi_right/Lx - 0.5*rho[i]*Lx/ep - phi_left/Lx
    # phi_analytic[i] = (0.5*rho[i]*x[i]**2)/ep + C_analytic*x[i] + phi_left
    
    ########### Solution for cn = constant, phi_left = 0.0, phi_right = 1.2 V
    #phi_analytic=((Z_cation+Z_anion)*cn*q_ele*x[i]**2)/(2*ep)-(((Lx**2*Z_cation+Lx**2*Z_anion)*cn*q_ele-2*ep*phi_right+2*ep*phi_left)*x)/(2*Lx*ep)+phi_left

    
    ########### Solution for cn_anion = cn*x/Lx, cn_cation = cn*(1-x/Lx)
    #Wphi_analytic[i] = -((Z_cation-Z_anion)*cn*q_ele*x[i]**3-3*Lx*Z_cation*cn*q_ele*x[i]**2)/(6*Lx*ep) - (((2*Lx**2*Z_cation+Lx**2*Z_anion)*cn*q_ele-6*ep*phi_right+6*ep*phi_left)*x[i])/(6*Lx*ep) + phi_left
    ############# Solution for cn_anion = cn_cation = cn(Lx-x)   ######################
    #phi_analytic[i] = -(Z_anion*cn*q_ele*x[i]**4+(2*Z_cation-2*Lx*Z_anion)*cn*q_ele*x[i]**3-6*Lx*Z_cation*cn*q_ele*x[i]**2)/(12*ep)- (((4*Lx**3*Z_cation+Lx**4*Z_anion)*cn*q_ele-12*ep*phi_right+12*ep*phi_left)*x[i])/(12*Lx*ep) - phi_left
    
    
    #######  Solution for cn(x) = cn*x*(Lx-x)/Lx^2
    phi_analytic[i] = ((Z_cation+Z_anion)*cn*q_ele*x[i]**4+(-2*Lx*Z_cation-2*Lx*Z_anion)*cn*q_ele*x[i]**3)/(12*Lx**2*ep) - (((Lx**2*Z_cation+Lx**2*Z_anion)*cn*q_ele-12*ep*phi_right+12*ep*phi_left)*x[i])/(12*Lx*ep)+phi_left



# phi_plot = np.zeros((Nx + 1))
# phi_plot = phi[int(Nt/2),:]
# phi_plot[0]= phi_left
# phi_plot[Nx] = phi_right

########## Potential Plots/Animaiton ##############################

it = 0
iplot = 0
phi_max = np.trunc(np.ceil(np.amax(phi)))*1.1
phi_min = np.floor(np.amin(phi))*1.1
print(phi_max, phi_min)
iframe = 1

while(it<=Nt):
    if(it ==0 or iplot == Ntplot):

        plt.rcParams.update({'font.size': 12})

        time = it*dt/3600
        Ttotal_hr = Ttotal/3600
        phi_plate = plt.figure(it, frameon=True, figsize=[6.0, 6.0],dpi = 300, constrained_layout= True)
        #phi_plate = plt.figure(it, frameon=True, dpi = 300, constrained_layout= True)

        ax = plt.subplot(111)
        #phi_plate.set_figwidth(3)
        plt.xlim(0, Lx)
        plt.ylim(phi_min, phi_max)
        plt.plot(x,phi[it,:], label = "numerical")
        plt.plot(x,phi_analytic, label = "analytic @ t=0.0 s", linestyle = '--')
        #plt.legend()
        
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1.17),
         fancybox=True, shadow=True, ncol=1)
        #ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=1)
        

        
        
        
        #plt.legend(bbox_to_anchor =(0.5,-0.5), loc='lower center')
        plt.xlabel("x [m]")
        plt.ylabel(r"$\phi~ [volts]$")
        plt.grid(visible=True, which="both", axis='both', color='black', linestyle='--', linewidth=1)         #plt.xlabel("L [m]", fontsize=18)
        phi_plate_name = "phi_plate_%04d.png" % (iframe)
        plt.title("Diffusion of NaCl in water \n t = %5.2f hr/ %5.2f hr \n $\phi_L = %3.1f~volts ~ \phi_R = %3.1f ~volts$ " %(time, Ttotal_hr, phi_left, phi_right), fontsize=12, loc="left")


        plt.savefig(phi_plate_name, dpi='figure', format='png', bbox_inches=None, pad_inches=0.5, transparent=False)
        iplot = 1
        iframe = iframe + 1

    it = it + 1
    iplot = iplot +1
    plt.ioff()
    plt.close(phi_plate)
    
cmd3 = "rm phi.webm"
os.system(cmd3)
cmd4 = "ffmpeg -hide_banner -loglevel panic -framerate 5 -i phi_plate_%04d.png phi.webm"
os.system(cmd4)
cmd5 = "rm phi_*.png"
os.system(cmd5)
    
##### Electric Field Plots/Animation ####################


#  Analytic Electric Field Calculation for Plotting 

Ex_analytic = np.zeros((Nx+1))

for i in np.arange(0, Nx+1):
    #x[i] = i*dx
    ########### Solution for cn = constant
    #Ex_analytic[i] = ((Z_cation+Z_anion)*cn*q_ele*x[i])/ep - ((Lx**2*Z_cation+Lx**2*Z_anion)*cn*q_ele-2*ep*phi_right+2*ep*phi_left)/(2*Lx*ep)
    
    
    ########### Solution for cn_anion = cn*x/Lx, cn_cation = cn*(1-x/Lx)
    #Wphi_analytic[i] = -((Z_cation-Z_anion)*cn*q_ele*x[i]**3-3*Lx*Z_cation*cn*q_ele*x[i]**2)/(6*Lx*ep) - (((2*Lx**2*Z_cation+Lx**2*Z_anion)*cn*q_ele-6*ep*phi_right+6*ep*phi_left)*x[i])/(6*Lx*ep) + phi_left
    ############# Solution for cn_anion = cn_cation = cn(Lx-x)   ######################
    #phi_analytic[i] = -(Z_anion*cn*q_ele*x[i]**4+(2*Z_cation-2*Lx*Z_anion)*cn*q_ele*x[i]**3-6*Lx*Z_cation*cn*q_ele*x[i]**2)/(12*ep)- (((4*Lx**3*Z_cation+Lx**4*Z_anion)*cn*q_ele-12*ep*phi_right+12*ep*phi_left)*x[i])/(12*Lx*ep) - phi_left
    
    ########### Solution for cn(x) = cn*x*(Lx-x)/Lx^2
    Ex_analytic[i] = -(4*(Z_cation+Z_anion)*cn*q_ele*x[i]**3+3*(-2*Lx*Z_cation-2*Lx*Z_anion)*cn*q_ele*x[i]**2)/(12*Lx**2*ep) + ((Lx**2*Z_cation+Lx**2*Z_anion)*cn*q_ele-12*ep*phi_right+12*ep*phi_left)/(12*Lx*ep)
    
#print(Ex_analytic)


######  Regular plot of Electric Field#######################3
it = 0
iplot = 0
Ex_max = np.ceil(np.amax(Ex))*1.1
Ex_min = np.floor(np.amin(Ex))*1.1
iframe = 0
while(it<=Nt):
    if(it ==0 or iplot == Ntplot):

        plt.rcParams.update({'font.size': 12})

        time = it*dt/3600
        Ttotal_hr = Ttotal/3600
        Ex_plate = plt.figure(it, frameon=True, figsize=[6.0, 6.0],dpi = 300, constrained_layout= True)
        ax = plt.subplot(111)
        plt.xlim(0, Lx)
        plt.ylim(Ex_min, Ex_max)
        #plt.ylim(-20, 20)
        plt.plot(x,Ex[it,:], label = "numerical")
        plt.plot(x,Ex_analytic[:], label = "analytic at t = 0.0 s", linestyle = '--')
        #plt.legend()
        #plt.legend(bbox_to_anchor =(0.5,-0.5), loc='lower center')
        
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1.17),
        fancybox=True, shadow=True, ncol=1)

        
        plt.xlabel("x [m]")
        plt.ylabel(r"$E_x~ [volts/m]$")
        plt.grid(visible=True, which="both", axis='both', color='black', linestyle='--', linewidth=1)         #plt.xlabel("L [m]", fontsize=18)
        Ex_plate_name = "Ex_plate_%04d.png" % (iframe)
        plt.title("Diffusion of NaCl in water \n t = %5.2f hr/ %5.2f hr \n $\phi_L = %3.1f~volts ~ \phi_R = %3.1f ~volts$ " %(time, Ttotal_hr, phi_left, phi_right), fontsize=12, loc = "left")

        plt.savefig(Ex_plate_name, dpi='figure', format='png', bbox_inches=None, pad_inches=0.5, transparent=False)
        iplot = 1
        iframe = iframe + 1

    it = it + 1
    iplot = iplot +1
    plt.ioff()
    plt.close(Ex_plate)



cmd3 = "rm Ex.webm"
os.system(cmd3)
cmd4 = "ffmpeg -hide_banner -loglevel panic -framerate 20 -i Ex_plate_%04d.png Ex.webm"
os.system(cmd4)
cmd5 = "rm Ex_*.png"
os.system(cmd5)

Ex_norm = np.amax(np.abs(Ex))
print('Exmax=',Ex_norm)

# ###########  Vector plot of Electric Fields [DOES NOT WORK]



# #Ey = np.array(0,1.0, num_vecs)

# it = 0
# iplot = 0
# iframe = 0
# while(it<=Nt):
#     if(it ==0 or iplot == Ntplot):

#         plt.rcParams.update({'font.size': 12})

#         time = it*dt/3600
#         Ttotal_hr = Ttotal/3600
#         vecEx_plate = plt.figure(it, frameon=True, figsize=[6.0, 6.0],dpi = 300, constrained_layout= True)
#         ax = plt.subplot(111)
#         plt.xlim(0, Lx)
#         #plt.ylim(Ex_min, Ex_max)
#         #plt.ylim(-20, 20)
#         #plt.plot(x,Ex[it,:], label = "numerical")
#         #plt.plot(x,Ex_analytic[:], label = "analytic at t = 0.0 s", linestyle = '--')
#         #plt.legend()
#         #plt.legend(bbox_to_anchor =(0.5,-0.5), loc='lower center')
        
        
#         plt.xlabel("x [m]")
#         plt.grid(visible=False)#, which="both", axis='both', color='black', linestyle='--', linewidth=1)         #plt.xlabel("L [m]", fontsize=18)
#         vecEx_plate_name = "vecEx_plate_%04d.png" % (iframe)
#         plt.title("Diffusion of NaCl in water \n t = %5.2f hr/ %5.2f hr \n $\phi_L = %3.1f~volts ~ \phi_R = %3.1f ~volts$ " %(time, Ttotal_hr, phi_left, phi_right), fontsize=12, loc = "left")

# #  Vector plot of Electric Fields



# #plt.subplot(2, 1, 1)
# # ax.set_xlabel('z [m]')
# # ax.set_ylabel('r [m]')
# # ax.set_title('Magnentic Field (B) vector Plot')
# # ax.grid(True, which='both', color='whitesmoke', linestyle=':',)
# # ax.set_ylim(0, R)
# # ax.set_xlim(0, L)
# # plt.quiver(znum, rnum, z_vector, r_vector, color='saddlebrown')
# # plt.quiver(znum, rnum, z_vector, r_vector, color='saddlebrown', pivot='middle')
#         num_skip = 5
#         find_size = np.size(Ex[it,::num_skip])
#         vecEx_plot = np.array((find_size))
#         vecEx_plot = Ex[it,::num_skip]/23


#         num_vecs = np.size(vecEx_plot)
#         #print('numvec =',find_size)
                
                
#         xplot = np.linspace(0, Lx, num_vecs)
#         yplot = np.linspace(0, 1.0, num_vecs)

#                 #y = np.linspace(0,1.0, Nx+1)
#         Ey = 0.0*np.linspace(0,1.0, num_vecs)


#         #Ey = np.zeros(np.size(Ex[it,:]))
#         #colors = [(255, 255, 255), (184, 115, 51), (139, 69, 19)] # This example uses the 8-bit RGB
#         #my_cmap = ListedColormap(colors)
#         #my_cmap = LinearSegmentedColormap(colors)
#         #my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white"])

#         # ax2=ax.twiny()
#         # #ax2.plot(x,y)
#         # ax2.ylim(0, 1.0)
#         #my_cmap = create_colormap(colors, bit=True)
#         #plt.quiver(Ex[it,:], Ey, x, y, Ex_norm, cmap=my_cmap, pivot='middle')
#         xplot,yplot = np.meshgrid(xplot,yplot)

#         vecEx_plot, Ey = np.meshgrid(vecEx_plot, Ey)
#         # fig, ax = plt.subplots()
#         # q = ax.quiver(x, y, Excurrent, Ey)
        
#         #plt.quiver(xplot,yplot,vecEx_plot, Ey, Ex_norm, cmap="RdBu", pivot='middle')
#         plt.quiver(xplot,yplot,vecEx_plot, Ey , color="firebrick", pivot='middle')
# #plt.quiver(coilz, coilr, z_vector, r_vector, color='saddlebrown', pivot='middle')


# # xi = np.linspace(coilz.min(), coilz.max(), 200)
# # yi = np.linspace(coilr.min(), coilr.max(), 200)
# # X, Y = np.meshgrid(xi, yi)
# # U = interpolate.griddata((coilz, coilr), Bz_ext, (X, Y), method='cubic')
# # V = interpolate.griddata((coilz, coilr), Br_ext, (X, Y), method='cubic')

# #plt.streamplot(X, Y, U, V, density=(1,0.5), color='saddlebrown', linewidth=2,  integration_direction='forward')
# #plt.streamplot(coilz, coilr, u,v , color='saddlebrown')

# #





#         plt.savefig(vecEx_plate_name, dpi='figure', format='png', bbox_inches=None, pad_inches=0.5, transparent=False)
#         iplot = 1
#         iframe = iframe + 1

#     it = it + 1
#     iplot = iplot +1
#     plt.ioff()
#     plt.close(vecEx_plate)



# cmd3 = "rm vecEx.webm"
# os.system(cmd3)
# cmd4 = "ffmpeg -hide_banner -loglevel panic -framerate 20 -i vecEx_plate_%04d.png vecEx.webm"
# os.system(cmd4)
# cmd5 = "rm vecEx_*.png"
# os.system(cmd5)


print("################################################################")




plt.plot(t/3600,m_anion, label = 'anion total mass')
plt.plot(t/3600,m_cation, label ='cation total mass', linestyle = '--')
plt.xlim(0, Ttotal/3600)
plt.ylim(0.0, 1)
plt.legend()

plt.xlabel("t [hr]")
plt.ylabel("mass ")

plt.show()

#play sound when done
finished_sound = os.path.dirname(os.path.realpath(__file__)) + '/' + 'paddansq.wav'
play_done_sound = subprocess.run(["play","-v","0.25", finished_sound])





