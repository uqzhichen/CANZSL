import torch
import torch.nn as nn

rdc_text_dim = 1000
z_dim = 100
h_dim = 4086
X_dim = 3072
text_dim = 11083

class _param:
    def __init__(self):
        self.rdc_text_dim = rdc_text_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

# reduce to dim of text first
class _netG(nn.Module):
    def __init__(self, text_dim=11083, X_dim=3072):
        super(_netG, self).__init__()
        self.rdc_text = nn.Linear(text_dim, rdc_text_dim)
        self.main = nn.Sequential(nn.Linear(z_dim + rdc_text_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())

    def forward(self, z, c):
        rdc_text = self.rdc_text(c)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output, rdc_text


#added class
"""
class _netC(nn.Module):
    def __init__(self, X_dim=3072, text_dim=11083, y_dim=150):
        super(_netC, self).__init__()
        self.rdc_text = nn.Linear(rdc_text_dim+z_dim, text_dim)

        self.main = nn.Sequential(nn.Linear(X_dim, h_dim),
                              nn.LeakyReLU(),
                              nn.Linear(h_dim, rdc_text_dim),
                              nn.Tanh())

        #Added
        self.C_aux = nn.Linear(rdc_text_dim,y_dim)
    def forward(self,z,input):
        text = self.main(input)
        noisy_text = torch.cat([z,text], 1)
        output = self.rdc_text(noisy_text)
        return output, self.C_aux(text)
"""
class _netD(nn.Module):
    def __init__(self, y_dim=150, X_dim=3072):
        super(_netD, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim),
                                      nn.ReLU())
        # Discriminator net branch one: For Gan_loss
        self.D_gan = nn.Linear(h_dim, 1)
        # Discriminator net branch two: For aux cls loss
        self.D_aux = nn.Linear(h_dim, y_dim)

    def forward(self, input):
        h = self.D_shared(input)

        return self.D_gan(h), self.D_aux(h)

class _netG2(nn.Module):
    def __init__(self, X_dim=3584):
        super(_netG2, self).__init__()
        self.main = nn.Sequential(nn.Linear(X_dim+z_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, rdc_text_dim))


    def forward(self,z, feature):
        input = torch.cat([z, feature], 1)
        rdc_text= self.main(input)
        return rdc_text


class _netD2(nn.Module):
    def __init__(self,y_dim=150):
        super(_netD2, self).__init__()
        self.D2_shared = nn.Sequential(nn.Linear(rdc_text_dim, h_dim),
                                      nn.ReLU())
        self.D2_gan = nn.Linear(h_dim,1)
        self.D2_aux = nn.Linear(h_dim, y_dim)


    def forward(self, rdc_text):
        h = self.D2_shared(rdc_text)
        return self.D2_gan(h), self.D2_aux(h)


# In GBU setting, using attribute
class _netG_att(nn.Module):
    def __init__(self, opt, att_dim, X_dim):
        super(_netG_att, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.z_dim + att_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())
    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        output = self.main(input)
        return output

class _netG2_att(nn.Module):
    def __init__(self, opt, att_dim, X_dim):
        super(_netG2_att, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.z_dim + X_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, att_dim))
    def forward(self, z, x):
        input = torch.cat([z, x], 1)
        output = self.main(input)
        return output

class _netD2_att(nn.Module):
    def __init__(self,att_dim,y_dim=150):
        super(_netD2_att, self).__init__()
        self.D2_shared = nn.Sequential(nn.Linear(att_dim, h_dim),
                                      nn.ReLU())

        self.D2_gan = nn.Linear(h_dim,1)
        self.D2_aux = nn.Linear(h_dim, y_dim)

    def forward(self, input):
        h = self.D2_shared(input)
        return self.D2_gan(h), self.D2_aux(h)