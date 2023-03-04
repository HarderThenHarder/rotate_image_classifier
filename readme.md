## 旋转验证码角度识别器

项目介绍：[这里](https://www.zhihu.com/zvideo/1512197159978483712)

项目中包含 **已训练好** 的模型，运行：

```python
python inference.py
```

得到对测试图片的角度预测：

<img src=''>

<table>
<tr>
<td><img src=img_examples/angle_136.png border=0></td>
<td><img src=img_examples/angle_323.png border=0></td>
<td><img src=img_examples/angle_287.png border=0></td>
</tr>
<tr>
<td><img src=img_examples/angle_236.png border=0></td>
<td><img src=img_examples/angle_92.png border=0></td>
<td><img src=img_examples/angle_234.png border=0></td>
</tr>
<tr>
<td><img src=img_examples/angle_202.png border=0></td>
<td><img src=img_examples/angle_349.png border=0></td>
<td><img src=img_examples/angle_177.png border=0></td>
</tr>
</table>


```python
Infer / Label: 138 / 136
Infer / Label: 323 / 323
Infer / Label: 283 / 287
Infer / Label: 236 / 236
Infer / Label: 90 / 92
Infer / Label: 234 / 234
Infer / Label: 202 / 202
Infer / Label: 353 / 349
Infer / Label: 176 / 177
Infer / Label: 45 / 49
Avg diff: 1.70
```