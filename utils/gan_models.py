import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import functools

IMG_W = IMG_H = 28  # image width and height
IMG_C = 1  # image channel


def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y:
    :param num_classes:
    :param dtype:
    :return:
    '''
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def pixel_norm(x, epsilon=1e-8):
    '''
    Pixel normalization
    :param x:
    :param epsilon:
    :return:
    '''
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdim=True) + epsilon)


class SpectralNorm(nn.Module):
    '''
    Spectral Normalization
    '''

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2_norm(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2_norm(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
        del u, v, w

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2_norm(u.data)
        v.data = l2_norm(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=False), upsample=functools.partial(F.interpolate, scale_factor=2)):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = SpectralNorm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = pixel_norm(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(pixel_norm(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Sigmoid(), latent_type='normal', img_h=28, img_w=28, img_c=1):
        super(GeneratorDCGAN, self).__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        assert self.img_w == self.img_h
        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, 4 * 4 * 4 * model_dim)
        deconv1 = nn.ConvTranspose2d(4 * model_dim, 2 * model_dim, 5)
        deconv2 = nn.ConvTranspose2d(2 * model_dim, model_dim, 5)
        deconv3 = nn.ConvTranspose2d(model_dim, self.img_c, 8, stride=2)
        # out_size = (in_size−1)×stride + dilation x(kernel_size−1) + 1

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact
        self.latent_type = latent_type

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)  # -> [B, z_dim + y_dim]
        output = self.fc(z_in)  # -> [B, 64 * model_dim]
        output = output.view(-1, 4 * self.model_dim, 4, 4)  # -> [B, 4 * model_dim, 4, 4]
        output = self.relu(output)
        output = pixel_norm(output)
        output = self.deconv1(output)  # -> [B, 2 * model_dim, 8, 8]
        output = output[:, :, :7, :7]  # -> [B, 2 * model_dim, 7, 7]
        output = self.relu(output)
        output = pixel_norm(output)
        output = self.deconv2(output)  # -> [B, model_dim, 11, 11]
        output = self.relu(output).contiguous()
        output = pixel_norm(output)
        output = self.deconv3(output)  # -> [B, img_c, 28, 28]
        output = self.outact(output)
        return output.view(-1, self.img_h * self.img_w * self.img_c)

    def sample_batch(self, batch_size, device):
        if self.latent_type == 'bernoulli':
            p = 0.5
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            z = bernoulli.sample((batch_size, self.z_dim)).view(batch_size, self.z_dim).to(device)
        else:
            z = torch.randn(batch_size, self.z_dim).to(device)
        y = torch.randint(0, self.num_classes, [batch_size]).to(device)
        return self.forward(z, y), y

    def sample(self, num_samples, device):
        images = []
        labels = []

        for i in range(num_samples // 100 + 1):
            img, lab = self.sample_batch(100, device)
            images.append(img)
            labels.append(lab)

        images = torch.cat(images)[:num_samples]
        labels = torch.cat(labels)[:num_samples]
        return images, labels


class GeneratorResNet(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Sigmoid()):
        super(GeneratorResNet, self).__init__()

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = SpectralNorm(nn.Linear(z_dim + num_classes, 4 * 4 * 4 * model_dim))
        block1 = GBlock(model_dim * 4, model_dim * 4)
        block2 = GBlock(model_dim * 4, model_dim * 4)
        block3 = GBlock(model_dim * 4, model_dim * 4)
        output = SpectralNorm(nn.Conv2d(model_dim * 4, IMG_C, kernel_size=3, padding=0))

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.fc = fc
        self.output = output
        self.relu = nn.ReLU()
        self.outact = outact

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, 4 * self.model_dim, 4, 4)
        output = self.relu(output)
        output = pixel_norm(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.outact(self.output(output))
        output = output[:, :, :-2, :-2]
        output = torch.reshape(output, [-1, IMG_H * IMG_W])
        return output


class DiscriminatorDCGAN(nn.Module):
    def __init__(self, model_dim=64, num_classes=10, if_SN=True):
        super(DiscriminatorDCGAN, self).__init__()

        self.model_dim = model_dim
        self.num_classes = num_classes

        if if_SN:
            self.conv1 = SpectralNorm(nn.Conv2d(1, model_dim, 5, stride=2, padding=2))
            self.conv2 = SpectralNorm(nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2))
            self.conv3 = SpectralNorm(nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2))
            self.linear = SpectralNorm(nn.Linear(4 * 4 * 4 * model_dim, 1))
            self.linear_y = SpectralNorm(nn.Embedding(num_classes, 4 * 4 * 4 * model_dim))
        else:
            self.conv1 = nn.Conv2d(1, model_dim, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2)
            self.linear = nn.Linear(4 * 4 * 4 * model_dim, 1)
            self.linear_y = nn.Embedding(num_classes, 4 * 4 * 4 * model_dim)
        self.relu = nn.ReLU()

    def forward(self, input, y):
        input = input.view(-1, 1, IMG_W, IMG_H)
        h = self.relu(self.conv1(input))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = h.view(-1, 4 * 4 * 4 * self.model_dim)
        out = self.linear(h)
        out += torch.sum(self.linear_y(y) * h, dim=1, keepdim=True)
        return out.view(-1)

    def calc_gradient_penalty(self, real_data, fake_data, y, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param y:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        y = y.to(device)
        alpha = torch.rand(batchsize, 1)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, y)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty
