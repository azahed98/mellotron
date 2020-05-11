from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out = model_output[0], model_output[1], model_output[2]
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

class TPSELoss(nn.Module):
    def __init__(self):
        super(TPSELoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, model_output):
        gst_embed, tpse_embed = model_output[4], model_output[5]
        gst_embed = gst_embed.detach().to('cuda:0')  # TODO: Fix for distributed GPU
        loss = self.loss(tpse_embed, gst_embed)
        return loss