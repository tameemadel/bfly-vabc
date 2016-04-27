import sys;
sys.path.append('/home/tameem/.spyder2')

import numpy as np
import pylab as pp
import scipy as sp
from scipy import special
#import tensorflow as tf
from abcpy.problems.exponential import *
from abcpy.plotting import *

#import autograd.numpy as np   # Thinly-wrapped version of Numpy
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.util import quick_grad_check

import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = StringIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))

def logweibull( x, lam, k ):
  log_pdf = np.log(k) - k*np.log(lam) + (k-1.0)*np.log(x) - pow( x / lam, k )
  return log_pdf
  
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def logofGaussian( x, q_mu, q_sigma ):
  log_pdf = -np.log(q_sigma * np.power(2 * np.pi, 0.5)) -  (np.power(x - q_mu,2) / (2.0 * np.power(q_sigma,2)))
  return log_pdf
  
epsilon = 0.25
n = 1
#M = 50
M = 181
nbr_parameters = 5
S = 1
  
def Q_0_nu( S, nbr_parameters ):
  #return np.random.rand( S,1 ).astype('float32')
  return np.random.rand( S, nbr_parameters ).astype('float32')

def Q_0_w( S, M ):
  return np.random.rand( S, M ).astype('float32')
  return np.random.rand( 2, S, M ).astype('float32')
  
#def calc_synth_data( w, g1_th01, g1_th02, g1_th03, g1_th04, g1_th05, tau, index):
def calc_synth_data( w, g1_th, tau, index):
    
    T = 180
    g1_th = np.exp(g1_th)       #TO BE REMOVED
    P = g1_th[0];
    delta = g1_th[1];
    N0 = g1_th[2];
    var_d = np.square(g1_th[3])
    prec_d = 1.0 / var_d
    
    var_p = np.square(g1_th[4])
    prec_p = 1.0 / var_p
    
    burnin = 50
    lag = int(np.floor(tau))
    if (float(tau)-float(lag)>0.5):
      lag = lag + 1

    N = np.zeros( lag+burnin+T, dtype=float)
    for i in range(lag):
        N[i] = 180
    

    for i in xrange(burnin+T):
      t = i + lag

      eps_t = gamma_rnd( prec_d.value, prec_d.value )
      e_t   = gamma_rnd( prec_p.value, prec_p.value )

      #tau_t = max(0,t-int(tau))
      tau_t = t - lag
      if ((t+2) < (burnin+T+lag)):
          N_bef = N[0:t]
          N_aft = N[t+2:]
          N_small_temp = np.array([P*N[tau_t]*np.exp(-N[tau_t]/N0)*e_t[0] + N[t-1]*np.exp(-delta*eps_t[0]), N[t+1]])
          N = np.concatenate((N_bef, N_small_temp, N_aft), axis=0)
      elif ((t+1) < (burnin+T+lag)):
          N_bef = N[0:t]
          N_small_temp = np.array([P*N[tau_t]*np.exp(-N[tau_t]/N0)*e_t[0] + N[t-1]*np.exp(-delta*eps_t[0]), N[t+1]])
          N = np.concatenate((N_bef, N_small_temp), axis=0)
      else:
          N_bef = N[0:t-1]
          N_small_temp = np.array([N[t-1], P*N[tau_t]*np.exp(-N[tau_t]/N0)*e_t[0] + N[t-1]*np.exp(-delta*eps_t[0])])
          N = np.concatenate((N_bef, N_small_temp), axis=0)
      
      
      #N[t] = P*N[tau_t]*np.exp(-N[tau_t]/N0)*e_t + N[t-1]*np.exp(-delta*eps_t)
    N_sorted = np.sort(N[-(T+1):])
    return N_sorted


blowfly_filename = "blowfly.txt"
observations   = np.loadtxt( blowfly_filename )[:,1]
sy = observations.mean()
sorted_observations = np.sort(observations)
sy1 = np.mean( sorted_observations[:observations.size/4] )
sy2 = np.mean( sorted_observations[observations.size/4:observations.size/2] )
sy3 = np.mean( sorted_observations[observations.size/2:3*observations.size/4] )
sy4 = np.mean( sorted_observations[3*observations.size/4:] )

def objective_fun_meth(q_log_theta):
    prior_alpha = 2.0
    prior_beta  = 0.5
    
    mean_prior  = prior_alpha/prior_beta
    var_prior   = prior_alpha/prior_beta**2
    
    phi_init_mean = np.log(mean_prior)
    phi_init_log_std = np.log(np.sqrt(var_prior))
    
    params = {}
    params["M"] = M
    params["sy"] = sy
    params["prior_alpha"] = prior_alpha
    params["prior_beta"] = prior_beta  # how many observations we draw per simulation
    params["fine_range"] = np.linspace(0.001,1.0,200)
    
    alpha = prior_alpha; beta = prior_beta;
    p = ExponentialProblem(alpha, beta)
    post_alpha = prior_alpha + M
    post_beta  = prior_beta + M*sy
    
    mean_post = post_alpha/post_beta
    var_post   = post_alpha/post_beta**2
    phi_post_mean = np.log(mean_post)
    phi_post_log_std = np.log(np.sqrt(var_post))
    
    #with tf.Session() as sess76:
    #for ii in range(0, 2):
    for ii in range(0, 1):
        count = 0;
        updater = count + 1;
    
        mu_log_P         = 2.0
        std_log_P        = 2.0
        mu_log_delta     = -1.0
        std_log_delta    = 2.0
        mu_log_N0        = 5.0
        std_log_N0       = 2.0
        mu_log_sigma_d   = 0.0
        std_log_sigma_d  = 2.0
        mu_log_sigma_p   = 0.0
        std_log_sigma_p  = 2.0
        mu_tau           = 15
        q_factor         = 0.1
        epsilon          = 10
        
        
        q_mu_log_theta = q_log_theta[0:5]
        q_log_sigma_log_theta = q_log_theta[5:]
        q_sigma_log_theta = np.exp(q_log_sigma_log_theta)
        
        
        pr_mu_log_P         = 2.0
        pr_sigma_log_P        = 2.0
        pr_mu_log_delta     = -1.0
        pr_sigma_log_delta    = 2.0
        pr_mu_log_N0        = 5.0
        pr_sigma_log_N0       = 2.0
        pr_mu_log_sigma_d   = 0.0
        pr_sigma_log_sigma_d  = 2.0
        pr_mu_log_sigma_p   = 0.0
        pr_sigma_log_sigma_p  = 2.0
        mu_tau           = 15
        q_factor         = 0.1
        epsilon          = 10
        
        #pr_mu_log_theta and  pr_sigma_log_theta are defined twice, here and before drawing the graphs. They will be unified into one definition
        pr_mu_log_theta = np.array([2.0, -1.0, 5.0, 0.0, 0.0])
        pr_sigma_log_theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        #pr_sigma_log_theta = np.array([0.0002, 0.0002, 0.0002, 0.0002, 0.0002])
                
        # place holders from samples from Q_0
        #nu = tf.placeholder("float", shape=[S, nbr_parameters], name="nu")
        #w = tf.placeholder("float", shape=[S, M], name="w")
        rs = npr.RandomState()
        nu = rs.random_sample((S, nbr_parameters))  #nu = rs.randn(S, nbr_parameters)
        w = rs.random_sample((S, M))
        
        #g1_theta = np.exp(-(np.square(np.log(nu) - q_mu_log_theta))/(2 * np.square(q_sigma_log_theta)))/(nu * q_sigma_log_theta * np.sqrt(2 * np.pi))
        g1_theta = q_mu_log_theta + q_sigma_log_theta * nu
        #g1_log_theta = np.log(g1_theta)
      
        T = 180
        mini_S = 1
        #x = tf.Variable(tf.zeros([mini_S, T+1]), name="x")
        x_temp = np.zeros([mini_S, T+1])
        #for index in range(mini_S):
        index = 0
        
        dd = calc_synth_data(w, g1_theta[index,:],poisson_rand(mu_tau),index)
        
        sx1 = np.mean( dd[:(T+1)/4] )
        sx2 = np.mean( dd[(T+1)/4:(T+1)/2] )
        #sx3 = tf.reduce_mean( dd[(T+1)/4:3*(T+1)/4,:] )
        sx3 = np.mean( dd[(T+1)/2:3*(T+1)/4] )
        sx4 = np.mean( dd[3*(T+1)/4:] )
        vx1 = np.mean(dd[:(T+1)/4]*dd[:(T+1)/4]) - sx1 * sx1
        vx2 = np.mean(dd[(T+1)/4:(T+1)/2]*dd[(T+1)/4:(T+1)/2]) - sx2 * sx2
        vx3 = np.mean(dd[(T+1)/2:3*(T+1)/4]*dd[(T+1)/2:3*(T+1)/4]) - sx3 * sx3
        vx4 = np.mean(dd[3*(T+1)/4:]*dd[3*(T+1)/4:]) - sx4 * sx4
        stdx1 = np.sqrt(vx1)
        stdx2 = np.sqrt(vx2)
        stdx3 = np.sqrt(vx3)
        stdx4 = np.sqrt(vx4)
        x = np.transpose(dd)
        sx = np.mean( x )
        vx = np.mean( x*x ) - sx*sx
        stdx = np.sqrt(vx)
                
        log_prior_theta = np.zeros([nbr_parameters], dtype=np.float32)
        log_prior_theta = -np.log(pr_sigma_log_theta * np.sqrt(2 * np.pi)) -  (np.square(g1_theta - pr_mu_log_theta) / (2.0 * np.square(pr_sigma_log_theta)))
            
        # prior for u is uniform(0,1)
        log_prior_u = 0.0
            
        #log_q_theta = tf.Variable(tf.zeros([S, nbr_parameters]), name = "log_q_theta")
        log_q_theta = -np.log(q_sigma_log_theta * np.power(2 * np.pi, 0.5)) -  (np.power(g1_theta - q_mu_log_theta,2) / (2 * np.power(q_sigma_log_theta,2)))
        
        eps_x = stdx/np.sqrt(M)
        eps_x1 = stdx1/np.sqrt(M)
        eps_x2 = stdx2/np.sqrt(M)
        eps_x3 = stdx3/np.sqrt(M)
        eps_x4 = stdx4/np.sqrt(M)
        #log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x)-0.5*tf.pow(sy - sx,2.0)/tf.pow(eps_x,2) )
        log_likelihood = np.mean( -0.5*np.sqrt(2*np.pi)-np.log(eps_x1)-0.5*np.power(sy1 - sx1,2.0)/np.power(eps_x1,2) ) #+\
        #np.mean( -0.5*np.sqrt(2*np.pi)-np.log(eps_x2)-0.5*np.power(sy2 - sx2,2.0)/np.power(eps_x2,2) )+\
        #np.mean( -0.5*np.sqrt(2*np.pi)-np.log(eps_x3)-0.5*np.power(sy3 - sx3,2.0)/np.power(eps_x3,2) )+\
        #np.mean( -0.5*np.sqrt(2*np.pi)-np.log(eps_x4)-0.5*np.power(sy4 - sx4,2.0)/np.power(eps_x4,2) )
        log_prior = np.mean( log_prior_theta )
        log_q     = np.mean( log_q_theta )
    
        # variational lower bound
        #vlb = log_likelihood + log_prior_1 + log_prior_2 + log_prior_4 + log_prior_5 - log_q_1 - log_q_2 - log_q_4 - log_q_5
        vlb = log_likelihood + log_prior - log_q
        #vlb = log_prior - log_q
        #vlb = log_likelihood + log_prior_1 - log_q_1
        #vlb = log_likelihood - log_q_1
        objective_function = -vlb / 1000
        
        print "========================================"
        print "1- check the prior is correct, why so off from true posterior? "
        print "2- kl divergence with true posterior"
        print "3- thetas from Q_phi(theta) have high prob under prior"
        print "4- mistakes converting gamma alpha/beta to lognormal means/vars"
        print "5- truncated normal instead of lognormal"
        print "6- variational for long tailed posterior, gamma?"
        print "7- non-rejection based generator for gamma? -- see wikipedia"
        print "========================================"
        
        return objective_function

loss_grad_obj = grad(objective_fun_meth)

q_log_theta = np.array([2.0, -1.0, 5.0, 0.0, 0.0, 0.6913, 0.6913, 0.6913, 0.6913, 0.6913])
#q_log_theta = np.array([2.669, -0.779, 6.545, 0.113, -0.2, -5.0, -5.0, -5.0, -5.0, -5.0])
q_log_theta = np.array([2.669, -0.779, 6.545, 0.113, -0.2, -1.0, -1.0, -1.0, -1.0, -1.0])
nbr_parameters = 5
log_prior_theta = np.zeros([nbr_parameters], dtype=np.float32)

for i in xrange(100):
    q_log_theta -= loss_grad_obj(q_log_theta) * 0.01



#curves
pr_mu_log_theta = np.array([2.0, -1.0, 5.0, 0.0, 0.0])
pr_sigma_log_theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
theta_mcmc = "theta_MCMC05.txt"
gtruth   = np.loadtxt( theta_mcmc )[:,0]
gtruth2   = np.loadtxt( theta_mcmc )[:,1]
gtruth3   = np.loadtxt( theta_mcmc )[:,2]
gtruth4   = np.loadtxt( theta_mcmc )[:,3]
gtruth5   = np.loadtxt( theta_mcmc )[:,4]


#----------------------------------
#'''
this_mu_log_delta = q_log_theta[0]
this_sigma_log_delta = np.exp(q_log_theta[5])
min_range = this_mu_log_delta - 3.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 3.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)

pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log P$')
#log_model_1 = logofGaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
model_1 = gaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, model_1, lw=2)
pp.legend( ["Posterior V-ABC", "true"])
#pp.show()

this_mu_log_delta_pr = pr_mu_log_theta[0]
this_sigma_log_delta_pr = pr_sigma_log_theta[0]
min_range_pr = this_mu_log_delta_pr - 3.5 * this_sigma_log_delta_pr
max_range_pr = this_mu_log_delta_pr + 3.5 * this_sigma_log_delta_pr
theta_range_pr = (min_range_pr, max_range_pr)
fine_theta_range_pr = np.arange(min_range_pr, max_range_pr+fine_bin_width, fine_bin_width)
model_2 = gaussian(fine_theta_range_pr, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range_pr, model_2, lw=2, ls='--')
pp.hist(gtruth, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper left', bbox_to_anchor=(0.51, 1.02), fancybox=True, shadow=True)
pp.show()
#pp.savefig("log_P_NIPS01.eps", format="eps")
pp.savefig("log_P_NIPS01.eps", format="eps")

pp.close('all')
'''
#----------------------------------
'''
this_mu_log_delta = q_log_theta[1]
this_sigma_log_delta = np.exp(q_log_theta[6])
min_range = this_mu_log_delta - 7.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 7.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log \delta$')
model_1 = gaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, model_1, lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()
#pp.savefig("foo.eps", format="eps")

this_mu_log_delta_pr = pr_mu_log_theta[1]
this_sigma_log_delta_pr = pr_sigma_log_theta[1]
min_range_pr = this_mu_log_delta_pr - 7.5 * this_sigma_log_delta_pr
max_range_pr = this_mu_log_delta_pr + 7.5 * this_sigma_log_delta_pr
theta_range_pr = (min_range_pr, max_range_pr)
fine_theta_range_pr = np.arange(min_range_pr, max_range_pr+fine_bin_width, fine_bin_width)
model_2 = gaussian(fine_theta_range_pr, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range_pr, model_2, lw=2, ls='--')
pp.hist(gtruth2, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper left', bbox_to_anchor=(0.51, 1.02), fancybox=True, shadow=True)
pp.show()
#pp.savefig("log_delta_final.eps", format="eps")
pp.savefig("log_delta_NIPS01.eps", format="eps")

pp.close('all')
'''
#----------------------------------
'''
this_mu_log_delta = q_log_theta[2]
this_sigma_log_delta = np.exp(q_log_theta[7])
min_range = this_mu_log_delta - 12.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 12.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log N_0$')
model_1 = gaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, model_1, lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()
#pp.savefig("foo.eps", format="eps")

this_mu_log_delta_pr = pr_mu_log_theta[2]
this_sigma_log_delta_pr = pr_sigma_log_theta[2]
min_range_pr = this_mu_log_delta_pr - 12.5 * this_sigma_log_delta_pr
max_range_pr = this_mu_log_delta_pr + 12.5 * this_sigma_log_delta_pr
theta_range_pr = (min_range_pr, max_range_pr)
fine_theta_range_pr = np.arange(min_range_pr, max_range_pr+fine_bin_width, fine_bin_width)
model_2 = gaussian(fine_theta_range_pr, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range_pr, model_2, lw=2, ls='--')
pp.hist(gtruth3, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper left', bbox_to_anchor=(0.01, 1.02), fancybox=True, shadow=True)
pp.show()
#pp.savefig("log_delta_final.eps", format="eps")
pp.savefig("log_N0_NIPS01.eps", format="eps")

pp.close('all')
'''
#----------------------------------
'''
this_mu_log_delta = q_log_theta[3]
this_sigma_log_delta = np.exp(q_log_theta[8])
min_range = this_mu_log_delta - 6.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 6.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log \sigma_d$')
model_1 = gaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, model_1, lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()
#pp.savefig("foo.eps", format="eps")

this_mu_log_delta_pr = pr_mu_log_theta[3]
this_sigma_log_delta_pr = pr_sigma_log_theta[3]
min_range_pr = this_mu_log_delta_pr - 6.5 * this_sigma_log_delta_pr
max_range_pr = this_mu_log_delta_pr + 6.5 * this_sigma_log_delta_pr
theta_range_pr = (min_range_pr, max_range_pr)
fine_theta_range_pr = np.arange(min_range_pr, max_range_pr+fine_bin_width, fine_bin_width)
model_2 = gaussian(fine_theta_range_pr, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range_pr, model_2, lw=2, ls='--')
pp.hist(gtruth4, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper left', bbox_to_anchor=(0.01, 1.02), fancybox=True, shadow=True)
pp.show()
#pp.savefig("log_delta_final.eps", format="eps")
pp.savefig("log_sigma_d_NIPS01.eps", format="eps")

pp.close('all')
'''
#----------------------------------
'''
this_mu_log_delta = q_log_theta[4]
this_sigma_log_delta = np.exp(q_log_theta[9])
min_range = this_mu_log_delta - 7.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 7.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log \sigma_p$')
model_1 = gaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, model_1, lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()
#pp.savefig("foo.eps", format="eps")

this_mu_log_delta_pr = pr_mu_log_theta[4]
this_sigma_log_delta_pr = pr_sigma_log_theta[4]
min_range_pr = this_mu_log_delta_pr - 7.5 * this_sigma_log_delta_pr
max_range_pr = this_mu_log_delta_pr + 7.5 * this_sigma_log_delta_pr
theta_range_pr = (min_range_pr, max_range_pr)
fine_theta_range_pr = np.arange(min_range_pr, max_range_pr+fine_bin_width, fine_bin_width)
model_2 = gaussian(fine_theta_range_pr, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range_pr, model_2, lw=2, ls='--')
pp.hist(gtruth5, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper left', bbox_to_anchor=(0.01, 1.02), fancybox=True, shadow=True)
pp.show()
#pp.savefig("log_delta_final.eps", format="eps")
pp.savefig("log_sigma_p_NIPS01.eps", format="eps")

pp.close('all')
#----------------------------------

#----------------------------------
#End of curves