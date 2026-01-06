

<!-- ![Alt text](assets/overview.jpg) -->

<table align="center">
  <tr>
    <td align="center"> 
      <img src="assets/methods.jpg" alt="Image 1" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> An overview of our PIAR.</em>
    </td>
  </tr>
</table>
</div>


### Details of our code.

1. Prepare data 
- Download raw data (as in paper we cited in dataset section) into `data\`, then run:
```
cd experiments/config/
chmod +x ./transform_data.sh
./transform_data.sh
```
3. Train PIAR model for each dataset, run:
```
cd experiments/config/
chmod +x ./train_piar.sh
./train_piar.sh
```
4. To test item-level attribution consistency, run:
```
cd experiments/config/
chmod +x ./test_attribution_robustness.sh
./test_attribution_robustness.sh
```
