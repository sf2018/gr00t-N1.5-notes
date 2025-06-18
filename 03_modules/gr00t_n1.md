## GR00T_N1类

**结构：**


backbone: EagleBackbone

action_head: FlowmatchingActionHead

prepare_input()

forward()

get_action()

## 1.构造函数

`config.backbone_cfg / config.action_head_cfg`：
    两个字典，分别初始化视觉-语言模型（EagleBackbone）和动作头（FlowmatchingActionHead）。
    
`action_horizon / action_dim`：
    控制动作序列的长度和维度。
    
`compute_dtype`：用于推理中数据类型控制（混合精度/float32 等）。

<pre><code>```
   def __init__(
        self,
        config: GR00T_N1Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)                # 检查 config 中的 backbone_cfg 和 action_head_cfg 是否都是字典。
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path                    # 模型路径，后续加载 checkpoint

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)  # 初始化动作预测模块

        self.action_horizon = config.action_horizon                # 保存动作序列的长度（时间长度）
        self.action_dim = config.action_dim                        # 保存动作维度
        self.compute_dtype = config.compute_dtype ```</code></pre>



## 2.forward 和 get_action

  def forward(self, inputs: dict) -> BatchFeature   训练阶段使用，返回包含 loss/action 等 key 的 BatchFeature
  
  def get_action(self, inputs: dict) -> BatchFeature  测试阶段使用，仅生成预测动作

<pre><code>‵‵‵
    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

```</code></pre>

<pre><code>‵‵‵
    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
```</code></pre>


  

## 3.prepare_input
def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]
将原始输入分别处理为 backbone 和 action head 的输入格式，通过 tree.map_structure 遍历并移动数据到模型设备，设置 dtype（仅 float 类型转换）

<pre><code>‵‵‵
 def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)    # 分别调用 backbone 和 action_head ，提取不同的模态
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs
```</code></pre>


## 4.from_pretrained
<pre><code>‵‵‵
def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) 加载预训练

   def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model
```</code></pre>
