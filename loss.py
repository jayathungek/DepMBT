from typing import List
import torch
import torch.nn as nn

from constants import *



def cross_entropy_loss_with_centering(teacher_output, student_output, centre):
    teacher_output = teacher_output.detach() # stop gradient
    s = nn.Sigmoid()(student_output / TEMP_STUDENT)
    t = nn.Sigmoid()((teacher_output - centre) / TEMP_TEACHER)
    return -(t * torch.log(s)).sum().mean()


def multicrop_loss(teacher_outputs: List[torch.tensor] , student_outputs: List[torch.tensor], centre: torch.tensor):
    total = torch.scalar_tensor(0).to(DEVICE)
    for teacher_view in teacher_outputs:
        for student_view in student_outputs:
            if not torch.equal(teacher_view, student_view):
                total += cross_entropy_loss_with_centering(teacher_view, student_view, centre)
    return total


if __name__ == "__main__":
    g1 = torch.rand(5)
    g2 = torch.rand(5)
    l1 = torch.rand(5)
    l2 = torch.rand(5)
    l3 = torch.rand(5)
    centre = torch.rand(5)


    t_vs = [g1, g2]
    s_vs = [g1, g2, l1, l2, l3]
    
    total_loss = multicrop_loss(t_vs, s_vs, centre)
    print(total_loss)

