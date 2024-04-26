import torch

class RELOBRALO():
    def __init__(self, num_losses: int, T: float = 0.1, alpha: float = 0.999,
                 rho: float = 0.999, device: str = 'cuda') -> None:
        '''
        Initialization of ReloBRaLo, a loss balancing scheme used to improve the convergence
        of Physics-Informed Neural Networks (PINNS) proposed by Rafael Bischof
        Input:
            num_losses: number of objectives/loss functions to be balanced
            T: temperature parameter
            rho: expected value of the bernoulli random value 'rho' used for exponential decay
            alpha: exponential decay parameter for the weights of the losses
        Output:
            None
        '''
        self.num_losses = num_losses
        self.l0 ={"l0_"+str(i): torch.tensor([1.], device=device) for i in range(num_losses)}
        self.lam ={"lam_"+str(i): torch.tensor([1.], device=device) for i in range(num_losses)}
        self.l ={"l_"+str(i): torch.tensor([1.], device=device) for i in range(num_losses)}
        self.T = torch.tensor(T, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, device=device)
        self.alpha = alpha

    def set_l0(self, loss_list: list[torch.Tensor]) -> None:
        '''This function is used to the save the values of the losses at the first epoch of
        training So it should be called as follows:
        ------------
        relobralo = RELOBRALO(num_losses)
        .
        .
        .
        for i in range(epochs):
            ### here you train the network then you compute the losses ###
            .
            .
            .
            if epoch == 0:
                relobralo.set_l0([list of losses(scalar values) in the trainng problem])
        .
        .
        .
        ------------
        Input:
            loss_list: the list of losses used in the multiobjective problem (torch scalars)
            NOTE! len(loss_list)==self.num_losses
        Output:
            None
        '''
        assert len(loss_list) == self.num_losses, 'Length of losses in the input list should\
             be equal to self.num_losses'
        assert all(isinstance(x, torch.Tensor) for x in loss_list), 'Each loss in loss_list\
             must be a salar tensor'

        for i in range(self.num_losses):
            self.l0['l0_'+str(i)] = loss_list[i].reshape(1)

    def __call__(self, loss_list: list[torch.Tensor]) -> torch.Tensor:
        '''This function returns the balanced loss of the muliobjective problem, where each loss
        in loss_list is multiplied by an adaptive weight.
        Call an instanciated object of this class after computing all the losses and pass a list of these losses
        Example:
        ------------
        relobralo = RELOBRALO(num_losses)
        for i in range(epochs):
            ### here you train the network then you compute the losses ###
            .
            .
            .
            loss_1 = MSE(y_hat, y)    # y_hat is one of the ouputs of the network
            loss_2 = MSE(residual, torch.zeros_like(residual))
            loss_3 ...
            # add any number of losses required
            balanced_loss = relobralo([loss_1, loss_2, loss_3, ...])
            balanced_loss.backward()
            optimizer.step()
            if epoch == 0 (or 1):
                relobralo.set_l0([list of losses(scalar values) in the trainng problem])
            .
            .
            .
        ------------
        Input:
            loss_list: the list of losses used in the multiobjective problem (torch scalars)
            NOTE: losses should be in the same order as when self
        Output:
            loss: torch scalar
        '''
        rho = torch.bernoulli(self.rho)
        lambs_hat = (torch.softmax(torch.cat([loss_list[i]/(self.l['l_'+str(i)]*self.T+1e-12) for i in range(self.num_losses)]),dim=0)*self.num_losses).detach()
        lambs0_hat = (torch.softmax(torch.cat([loss_list[i]/(self.l0['l0_'+str(i)]*self.T+1e-12) for i in range(self.num_losses)]),dim=0)*self.num_losses).detach()
        lambs = [rho*self.alpha*self.lam['lam_'+str(i)] + (1-rho)*self.alpha*lambs0_hat[i] + (1-self.alpha)*lambs_hat[i] for i in range(self.num_losses)]
        loss = torch.sum(torch.cat([lambs[i]*loss_list[i] for i in range(self.num_losses)]))
        for i in range(self.num_losses):
            self.lam['lam_'+str(i)] = lambs[i]
            self.l['l_'+str(i)] = loss_list[i].reshape(1)
        return loss
