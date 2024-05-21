import torch 
from typing import Dict, List, Tuple

from fcore.utils.structure import GenericData

TTensorList = List[torch.Tensor]
TTensorDict = Dict[str, torch.Tensor]
TTensorTuple = Tuple[torch.Tensor]
TTensor = torch.Tensor
TDataList = List[GenericData]
TData = GenericData