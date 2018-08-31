# gan-with-fake-id

Generative adversarial network 을 활용해서 Fake ID를 생성하는 실험입니다.


# dataset

다음과 같은 `real_ids.csv` 파일이 있어야 합니다.

```
아이디1
아이디2
...
아이디N
```



# 의존 라이브러리

* numpy

* matplotlib

* tensorflow (gpu)


# 실행

```
$ python3 fake-id-gan.py
```


----

# 결과

```
session loaded
graph variable initialized
training start at 1535696490.418216
total real ids: 15439
epoch: 00000  lossG:      0.922  lossD:      2.071  learning_rate: 0.0010  fake_id: 56jm7mm.fm85phxxinn4ef_4gquteqav5d20jel
epoch: 00020  lossG:      1.387  lossD:      1.389  learning_rate: 0.0010  fake_id: qjaagobl_ysei_iayeam_tiisnentyou1_oexo
epoch: 00040  lossG:      1.388  lossD:      1.389  learning_rate: 0.0010  fake_id: 4nu2o8mjleukemei.opaeenine___ennbmygeso
epoch: 00060  lossG:      1.360  lossD:      1.424  learning_rate: 0.0010  fake_id: b2sne1dke_s_yj_e_ruwxhcjienkpiiaa_il_iey
epoch: 00080  lossG:      1.371  lossD:      1.404  learning_rate: 0.0010  fake_id: b0_nnhtm6mqi6tyaky9on7nj_mnyioio2e24_g_3
epoch: 00100  lossG:      1.413  lossD:      1.360  learning_rate: 0.0010  fake_id: 4lccgrylnuaaooenlieslt_noosyeyn_t2c29i63
epoch: 00120  lossG:      1.400  lossD:      1.376  learning_rate: 0.0010  fake_id: atlmindgyboliey_auosoonae__nee_ze__n9i5s
epoch: 00140  lossG:      7.021  lossD:      0.723  learning_rate: 0.0010  fake_id: 40mcjl_t0lg0rytuaeraoonelr_2unig9_lenega
epoch: 00160  lossG:      1.363  lossD:      1.412  learning_rate: 0.0010  fake_id: unywnryur_tj7n6712hdinooonuo_oazaoe2iu5u
epoch: 00180  lossG:      1.405  lossD:      1.369  learning_rate: 0.0010  fake_id: fleo_yo_yeo_ae_aen_59ut0aauneyjoq0i2ud_d
epoch: 00200  lossG:      1.366  lossD:      1.408  learning_rate: 0.0010  fake_id: 3e1nmon___d_ileo__u5_iuny7ihiueenoy29g5z
epoch: 00220  lossG:      5.232  lossD:      0.727  learning_rate: 0.0010  fake_id: j2muhh_1yu2satwixoisnini_ei_1_yieang0b51
```
