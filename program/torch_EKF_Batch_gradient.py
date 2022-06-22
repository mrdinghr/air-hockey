import torch
import torch_air_hockey_baseline
from torch_EKF_Wrapper import air_hockey_EKF
from matplotlib import pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda")
table_length = 1.948
data_after_clean = np.load('total_data_after_clean.npy', allow_pickle=True)
train_data = data_after_clean[10:15]
test_data = data_after_clean[6:7]
# test_index = np.random.randint(0, len(data_after_clean), size=2)
# test_index = 2
# test_data = data_after_clean[2:3]
# train_data = np.ma.array(data_after_clean, mask=False)
# train_data.mask[test_index] = True
torch.set_printoptions(precision=8)


def calculate_init_state(data):
    state_dx = ((data[1][0] - data[0][0]) / (data[1][3] - data[0][3]) + (
            data[2][0] - data[1][0]) / (
                        data[2][3] - data[1][3]) + (data[3][0] - data[2][0]) / (
                        data[3][3] - data[2][3])) / 3
    state_dy = ((data[1][1] - data[0][1]) / (data[1][3] - data[0][3]) + (
            data[2][1] - data[1][1]) / (
                        data[2][3] - data[1][3]) + (data[3][1] - data[2][1]) / (
                        data[3][3] - data[2][3])) / 3
    state_dtheta = ((data[1][2] - data[0][2]) / (data[1][3] - data[0][3]) + (
            data[2][2] - data[1][2]) / (
                            data[2][3] - data[1][3]) + (data[3][2] - data[2][2]) / (
                            data[3][3] - data[2][3])) / 3
    state = torch.tensor([data[1][0], data[1][1], state_dx, state_dy, data[1][2], state_dtheta], device=device)
    return state


class EKFGradient(torch.nn.Module):
    def __init__(self, params, covariance_params):
        super(EKFGradient, self).__init__()
        self.register_parameter('dyna_params', torch.nn.Parameter(params))
        self.covariance_params = covariance_params
        # self.register_parameter('covariance_params', covariance_params)
        self.system = torch_air_hockey_baseline.SystemModel(tableDamping=self.dyna_params[1],
                                                            tableFriction=self.dyna_params[0],
                                                            tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                            puckRadius=0.03165, malletRadius=0.04815,
                                                            tableRes=self.dyna_params[3],
                                                            malletRes=0.8, rimFriction=self.dyna_params[2], dt=1 / 120)
        self.table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25,
                                                              puckRadius=0.03165,
                                                              restitution=self.dyna_params[2],
                                                              rimFriction=self.dyna_params[3], dt=1 / 120)
        self.R = torch.zeros((3, 3), device=device)
        self.R[0][0] = self.covariance_params[0]
        self.R[1][1] = self.covariance_params[1]
        self.R[2][2] = self.covariance_params[2]
        self.Q = torch.zeros((6, 6), device=device)
        self.Q[0][0] = self.covariance_params[3]
        self.Q[1][1] = self.covariance_params[3]
        self.Q[2][2] = self.covariance_params[4]
        self.Q[3][3] = self.covariance_params[4]
        self.Q[4][4] = self.covariance_params[5]
        self.Q[5][5] = self.covariance_params[6]
        self.P = torch.eye(6, device=device) * 0.01
        self.puck_EKF = air_hockey_EKF(u=1 / 120., system=self.system, table=self.table, Q=self.Q, R=self.R, P=self.P)

    '''
    input: list of trajectories 
    size: n*m*4 n:num of trajectories m:num of points for each trajectory  4: x y theta time
    output: list of loss for each point size: 1 dim and len = all points that can be used to update EKF
    '''

    def make_loss_list(self, data_set):
        length = len(data_set)
        # loss_list = torch.tensor([0], device=device)
        loss_list = []
        for i_trajectory in range(length):
            cur_trajectory = data_set[i_trajectory]
            cur_trajectory = torch.tensor(cur_trajectory, device=device).float()
            self.puck_EKF.refresh(self.P, self.Q, self.R)
            self.puck_EKF.init_state(calculate_init_state(cur_trajectory))
            i = 0
            j = 1
            while j < len(cur_trajectory)-1:
                i = i + 1
                self.puck_EKF.predict()
                # loss_list.append(self.puck_EKF.predict_state[:2].sum())
                # self.puck_EKF.state = self.puck_EKF.predict_state.clone()
                # j += 1
                if (i - 0.2) / 120 < cur_trajectory[j + 1][-1] - cur_trajectory[1][-1] < (
                        i + 0.2) / 120:
                    self.puck_EKF.update(cur_trajectory[j + 1][0:3])
                    j = j + 1
                    sign, logdet = torch.linalg.slogdet(self.puck_EKF.S)
                    cur_point_loss = sign * torch.exp(logdet) + self.puck_EKF.y @ torch.linalg.inv(
                        self.puck_EKF.S) @ self.puck_EKF.y
                    # loss_list.append(cur_point_loss.clone())
                    loss_list.append(cur_point_loss)
                elif cur_trajectory[j + 1][-1] - cur_trajectory[1][-1] <= (i - 0.2) / 120:
                    j = j + 1
                    self.puck_EKF.state = self.puck_EKF.predict_state
                else:
                    self.puck_EKF.state = self.puck_EKF.predict_state

                self.puck_EKF.predict_state = None
                self.puck_EKF.P.detach_()
                self.puck_EKF.F.detach_()
                self.puck_EKF.y = None
                self.puck_EKF.S = None
        return loss_list[1:]


if __name__ == '__main__':
    # table friction, table damping, table restitution, rim friction
    init_params = torch.Tensor([0.125, 0.375, 0.675, 0.145])
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1])
    model = EKFGradient(init_params, covariance_params)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Batch_size = 20
    writer = SummaryWriter('./bgd')
    epoch = 0
    for t in range(100):
        print(str(t)+' epoch')
        optimizer.zero_grad()
        loss_list = model.make_loss_list(train_data)
        # dataset_loss = Data.TensorDataset(loss_list)
        loader = Data.DataLoader(loss_list, batch_size=Batch_size, shuffle=True)
        for loss_batch in loader:
            optimizer.zero_grad()
            sum_loss_batch = torch.mean(loss_batch)
            sum_loss_batch.backward(retain_graph=True)
            # writer.add_scalar('loss of batch', sum_loss_batch.data, epoch)
            # writer.add_scalar('table damping batch', model.dyna_params[1], epoch)
            # writer.add_scalar('table friction batch', model.dyna_params[0], epoch)
            # writer.add_scalar('table restitution batch', model.dyna_params[2], epoch)
            # writer.add_scalar('rim friction batch', model.dyna_params[3], epoch)
            print(str(epoch)+' loss '+str(sum_loss_batch))
            print('params '+str(model.get_parameter('dyna_params').data))
            print('grad '+str(model.get_parameter('dyna_params').grad))
            optimizer.step()
            for p in model.get_parameter('dyna_params'):
                p.data.clamp_(0, 1)
            epoch += 1
        writer.add_scalar('loss of train set', sum(loss_list)/len(loss_list), t)
        writer.add_scalar('table damping', model.dyna_params[1], t)
        writer.add_scalar('table friction', model.dyna_params[0], t)
        writer.add_scalar('table restitution', model.dyna_params[2], t)
        writer.add_scalar('rim friction', model.dyna_params[3], t)
        test_loss_list = model.make_loss_list(test_data)
        writer.add_scalar('loss of test set', sum(test_loss_list)/len(test_loss_list), t)
    writer.close()
