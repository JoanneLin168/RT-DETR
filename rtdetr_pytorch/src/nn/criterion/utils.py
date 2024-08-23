import torch 
import torchvision



def format_target(targets):
    '''
    Args:
        targets (List[Dict]),
    Return: 
        tensor (Tensor), [im_id, label, bbox, mask]
    '''
    outputs = []
    for i, tgt in enumerate(targets):
        boxes =  torchvision.ops.box_convert(tgt['boxes'], in_fmt='xyxy', out_fmt='cxcywh') 
        labels = tgt['labels'].reshape(-1, 1)
        masks = tgt['masks']
        im_ids = torch.ones_like(labels) * i
        outputs.append(torch.cat([im_ids, labels, boxes, masks], dim=1))

    return torch.cat(outputs, dim=0)
