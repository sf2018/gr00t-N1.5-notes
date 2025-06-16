## GR00T_N1类

**结构：**


GR00T_N1
backbone: EagleBackbone
action_head: FlowmatchingActionHead
prepare_input()
forward()
get_action()

1.构造函数

config.backbone_cfg / config.action_head_cfg：
    两个字典，分别初始化视觉-语言模型（EagleBackbone）和动作头（FlowmatchingActionHead）。
action_horizon / action_dim：
    控制动作序列的长度和维度。
compute_dtype：用于推理中数据类型控制（混合精度/float32 等）。

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






  

2.forward 和 get_action
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



  

3.prepare_input
def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]
将原始输入分别处理为 backbone 和 action head 的输入格式，通过 tree.map_structure 遍历并移动数据到模型设备，设置 dtype（仅 float 类型转换）

4.from_pretrained
def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) 加载预训练
