class LAMB(Optimizer):
    def __init__(self, lr, lamb=0.01,beta_1=0.9,
                 beta_2=0.999, eps=1e-6, **kwargs):
        super(LAMB, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, dtype='float32', name='lr')
            self.lamb =  K.variable(lamb, dtype='float32', name='lambda')
            self.beta_1 = K.variable(beta_1, dtype='float32', name='beta1')
            self.beta_2 = K.variable(beta_2, dtype='float32', name='beta2')
            self.eps= K.variable(eps, dtype='float32', name='eps')
    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'lambda': float(K.get_value(self.lamb)),
                  'beta1': float(K.get_value(self.beta1)),
                  'beta2': float(K.get_value(self.beta2)),
                  'eps':float(K.get_value(self.eps)),
                  }
        base_config = super(LAMB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))                
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
       # weights = self.get_weights()
        self.updates = [K.update_add(self.iterations, 1)]

        # momentum
        ms = [K.zeros(K.int_shape(param), dtype=K.dtype(param))
                   for param in params]
        vs = [K.zeros(K.int_shape(param), dtype=K.dtype(param))
                   for param in params]
        self.weights  = [self.iterations] + ms + vs
        for p, g, m,v  in zip(params, grads, ms,vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            mhat_t = m_t /(1 - self.beta_1)
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            vhat_t = v_t/(1-self.beta_2)
            self.updates.append(K.update(m,m_t))
            self.updates.append(K.update(v,v_t))
            r1 = K.l2_normalize(p)#, prevweights find example
            r2 = K.l2_normalize((mhat_t / K.pow(vhat_t+self.eps,0.5)) + self.lamb*p)
            r = r1/r2
            lr_batch=  r*self.lr
      
            new_p = p - lr_batch*((mhat_t / K.pow(vhat_t+self.eps,0.5))+ self.lamb*p)
            if getattr(p, 'constraint', None) is not None:
                        new_p= p.constraint(new_p)
            self.updates.append(K.update(p,new_p))
        return self.updates
