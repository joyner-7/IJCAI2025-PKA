from __future__ import absolute_import
from collections import OrderedDict
import torch

from ..utils import to_torch

def extract_cnn_feature(model,inputs):#,training_phase=None,fkd=True
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).cuda()
        
        Expand=False
        if inputs.size(0)<2:
            Pad=inputs[:1]
            inputs=torch.cat((inputs,Pad),dim=0)
            Expand=True

        #print('inputs.size() =',inputs.size())
        outputs = model(inputs)[0]#,training_phase=training_phase
        #outputs = outputs.data.cpu()
        outputs = outputs.detach().to(inputs.device)

        if Expand:
            outputs=outputs[:-1]

        return outputs

