// UTF-8

data {
  int<lower=1> N;                
  int<lower=1> N_month;          
  int<lower=1> N_age;            
  int<lower=1, upper=N_month> month[N];  
  int<lower=1, upper=N_age> age[N];      
  int<lower=0, upper=1> t[4, N];   
  int<lower=-1, upper=1> y[4, N];  
  real<lower=-1, upper=1> W[4, N];   
  real<lower=0, upper=1> Sp[4];  
}

parameters {
  matrix<lower=0, upper=1>[N_month, N_age] p_ma;  
  vector<lower=0, upper=1>[4] Se;                
}


transformed parameters {

  vector[N] log_p_z1; 
  vector[N] log_p_z0; 
  
  for (i in 1:N) { 

    int m = month[i];
    int a = age[i];
    
    
    real lp_z1 = log(p_ma[m, a] + 1e-10);  
    real lp_z0 = log(1 - p_ma[m, a] + 1e-10);  
    
    for (j in 1:4) {
      if (t[j, i] == 1 && y[j, i] != -1) { 
      real prob_z1 = fmax(1e-5, fmin(1.0-1e-5, Se[j] * W[j, i])); 
      lp_z1 += bernoulli_lpmf(y[j, i] | prob_z1);
      lp_z0 += bernoulli_lpmf(y[j, i] | (1 - Sp[j])); 
      }
    }
    log_p_z1[i] = lp_z1;
    log_p_z0[i] = lp_z0;
  }
}


model {
  to_vector(p_ma) ~ beta(1, 1); 
  Se ~ beta(1, 1);  

  for (i in 1:N) {
    target += log_sum_exp(log_p_z1[i], log_p_z0[i]); 
  }
}
