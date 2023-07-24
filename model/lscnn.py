from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.bn_1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_2 = nn.LayerNorm(hidden_dim)
        self.linear_branch = nn.Linear(in_dim, hidden_dim)
        self.bn_branch = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.bn_1(self.linear_1(x))
        y = self.relu(y)
        y = self.bn_2(self.linear_2(y))
        y_branch = self.bn_branch(self.linear_branch(x))

        return self.relu(y + y_branch), y


class LSCNN(nn.Module):
    def __init__(self, is_project=False):
        super(LSCNN, self).__init__()
        # PD branch
        self.res_pd1 = ResidualBlock(in_dim=122, hidden_dim=256)
        self.res_pd2 = ResidualBlock(in_dim=256, hidden_dim=512)
        self.sequential_pd = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        # FD branch
        self.res_fd1 = ResidualBlock(in_dim=1024, hidden_dim=256)
        self.res_fd2 = ResidualBlock(in_dim=256, hidden_dim=512)
        self.sequential_fd = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
        self.relu = nn.ReLU()

        self.is_project = is_project
        if is_project:
            self.project = nn.Linear(512, 512)

    def forward(self, fiture):
        # pd
        pd = fiture[:, 0:122]
        pd, _ = self.res_pd1(pd)
        pd, latent_rep_pd = self.res_pd2(pd)
        pd = self.sequential_pd(pd)
        # fd
        fd = fiture[:, 122:]
        fd, _ = self.res_fd1(fd)
        fd, latent_rep_fd = self.res_fd2(fd)
        fd = self.sequential_fd(fd)

        if self.is_project:
            latent_rep_pd = self.project(latent_rep_pd)
            latent_rep_fd = self.project(latent_rep_fd)

        return pd, fd, latent_rep_pd, latent_rep_fd
