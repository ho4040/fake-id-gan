# fake id 생성기

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
training start at 1536307785.287672
total real ids: 15439
epoch: 00000  l_G: 1.411   l_D: 1.364   a: 0.0010    fake_sample: 24b_eyaoo0vin1iak1
epoch: 00001  l_G: 1.342   l_D: 1.439   a: 0.0010    fake_sample: hyu.6iknkkt_a_6
epoch: 00002  l_G: 1.367   l_D: 1.408   a: 0.0010    fake_sample: h03udnaalau8_hn.19fgy3ija1j
epoch: 00003  l_G: 1.365   l_D: 1.409   a: 0.0010    fake_sample: 3h_slfrdip0bix1e.
epoch: 00004  l_G: 1.324   l_D: 1.463   a: 0.0010    fake_sample: _jsrp_sy
epoch: 00005  l_G: 1.356   l_D: 1.417   a: 0.0010    fake_sample: eetfat.inl3gbtw
epoch: 00006  l_G: 1.372   l_D: 1.402   a: 0.0010    fake_sample: azak75mn9tmen_u_0uo291y2
epoch: 00007  l_G: 1.372   l_D: 1.402   a: 0.0010    fake_sample: _o_sk.ns_g
epoch: 00008  l_G: 1.385   l_D: 1.388   a: 0.0010    fake_sample: ennu_kt_oipkyab
epoch: 00009  l_G: 1.370   l_D: 1.403   a: 0.0010    fake_sample: _mu7emamtutb8eenm_0ts_olrresa
epoch: 00010  l_G: 1.377   l_D: 1.400   a: 0.0010    fake_sample: xo_uvaxur
epoch: 00011  l_G: 1.381   l_D: 1.392   a: 0.0010    fake_sample: megi2snyiei.ea
epoch: 00012  l_G: 1.381   l_D: 1.392   a: 0.0010    fake_sample: a2hb_uhnuev2aijn7n
epoch: 00013  l_G: 1.372   l_D: 1.401   a: 0.0010    fake_sample: .8oby8o8aoraon1_y_
epoch: 00014  l_G: 1.369   l_D: 1.406   a: 0.0010    fake_sample: 0edleeynl0aarmbl0h2o
epoch: 00015  l_G: 1.390   l_D: 1.383   a: 0.0010    fake_sample: n4vytitjkg_meosj
epoch: 00016  l_G: 1.362   l_D: 1.412   a: 0.0010    fake_sample: nwnuoyien0ui
epoch: 00017  l_G: 1.385   l_D: 1.388   a: 0.0010    fake_sample: j1_mehimdlcon
epoch: 00018  l_G: 1.351   l_D: 1.425   a: 0.0010    fake_sample: 5lilmryigs_0ln
epoch: 00019  l_G: 1.434   l_D: 1.629   a: 0.0010    fake_sample: i6_i1
epoch: 00020  l_G: 1.861   l_D: 1.187   a: 0.0010    fake_sample: degolio
epoch: 00021  l_G: 1.116   l_D: 1.939   a: 0.0010    fake_sample: soen3mn33n
epoch: 00022  l_G: 1.457   l_D: 1.343   a: 0.0010    fake_sample: yby8hneho
epoch: 00023  l_G: 1.711   l_D: 1.132   a: 0.0010    fake_sample: rce12_seie
epoch: 00024  l_G: 1.308   l_D: 1.505   a: 0.0010    fake_sample: b___rer1d_hc
epoch: 00025  l_G: 1.456   l_D: 1.340   a: 0.0010    fake_sample: 6.n__i0yrdj9n6h0.
epoch: 00026  l_G: 1.396   l_D: 1.391   a: 0.0010    fake_sample: baa5e_ah_e1odeis
epoch: 00027  l_G: 1.324   l_D: 1.478   a: 0.0010    fake_sample: tliwarr1nl
epoch: 00028  l_G: 1.431   l_D: 1.345   a: 0.0010    fake_sample: lmelnhgh_bn3
epoch: 00029  l_G: 1.365   l_D: 1.411   a: 0.0010    fake_sample: _ih10_xenncck9
epoch: 00030  l_G: 1.375   l_D: 1.407   a: 0.0010    fake_sample: ok7ngr_s_nlasnsg.
epoch: 00031  l_G: 1.376   l_D: 1.399   a: 0.0010    fake_sample: hni1aa_un3e
epoch: 00032  l_G: 1.364   l_D: 1.419   a: 0.0010    fake_sample: dyknpsny_r6
epoch: 00033  l_G: 1.325   l_D: 1.466   a: 0.0010    fake_sample: 3a22gyaalodcngenn
epoch: 00034  l_G: 1.363   l_D: 1.414   a: 0.0010    fake_sample: enkmjh4sulryoaj
epoch: 00035  l_G: 1.352   l_D: 1.426   a: 0.0010    fake_sample: wiuamekon
epoch: 00036  l_G: 1.357   l_D: 1.419   a: 0.0010    fake_sample: o0uswljxsmiku
epoch: 00037  l_G: 1.408   l_D: 1.366   a: 0.0010    fake_sample: nh2boiabmodeoso4
epoch: 00038  l_G: 1.400   l_D: 1.374   a: 0.0010    fake_sample: jrrar2h_eeamh
epoch: 00039  l_G: 1.326   l_D: 1.459   a: 0.0010    fake_sample: enuna_fmi5
epoch: 00040  l_G: 1.349   l_D: 1.425   a: 0.0010    fake_sample: eswehvoonz__o
epoch: 00041  l_G: 1.388   l_D: 1.388   a: 0.0010    fake_sample: ysys.t1ew1v
epoch: 00042  l_G: 1.379   l_D: 1.395   a: 0.0010    fake_sample: hryri0iatnj
epoch: 00043  l_G: 1.386   l_D: 1.389   a: 0.0010    fake_sample: 2rrahii2adr2n5
epoch: 00044  l_G: 1.389   l_D: 1.384   a: 0.0010    fake_sample: _lnbbejrs_nyone
epoch: 00045  l_G: 1.383   l_D: 1.390   a: 0.0010    fake_sample: cos___io_
epoch: 00046  l_G: 1.388   l_D: 1.386   a: 0.0010    fake_sample: esung_oyyjar
epoch: 00047  l_G: 1.381   l_D: 1.393   a: 0.0010    fake_sample: hoiex11myo8hn7
epoch: 00048  l_G: 1.358   l_D: 1.415   a: 0.0010    fake_sample: acmwru
epoch: 00049  l_G: 1.394   l_D: 1.378   a: 0.0010    fake_sample: ha_0nrl_o
epoch: 00050  l_G: 1.387   l_D: 1.386   a: 0.0010    fake_sample: hjtkhd_rez.
epoch: 00051  l_G: 1.384   l_D: 1.389   a: 0.0010    fake_sample: hynai_vma
epoch: 00052  l_G: 1.372   l_D: 1.401   a: 0.0010    fake_sample: .nemiywtu
epoch: 00053  l_G: 1.385   l_D: 1.388   a: 0.0010    fake_sample: eie1_oee7u_0
epoch: 00054  l_G: 1.376   l_D: 1.397   a: 0.0010    fake_sample: desoa0n
epoch: 00055  l_G: 1.385   l_D: 1.388   a: 0.0010    fake_sample: bbaj_j_
epoch: 00056  l_G: 1.394   l_D: 1.379   a: 0.0010    fake_sample: rguesa.v_2
epoch: 00057  l_G: 1.384   l_D: 1.388   a: 0.0010    fake_sample: nhhasei
epoch: 00058  l_G: 1.386   l_D: 1.387   a: 0.0010    fake_sample: np_hysico_e3
epoch: 00059  l_G: 1.388   l_D: 1.384   a: 0.0010    fake_sample: cro_ujnnb46tkchgy_
epoch: 00060  l_G: 1.378   l_D: 1.395   a: 0.0010    fake_sample: wey_
epoch: 00061  l_G: 1.383   l_D: 1.389   a: 0.0010    fake_sample: maamuajiiu___
epoch: 00062  l_G: 1.380   l_D: 1.393   a: 0.0010    fake_sample: m_nouef
epoch: 00063  l_G: 1.404   l_D: 1.369   a: 0.0010    fake_sample: _1lef3_si
epoch: 00064  l_G: 1.386   l_D: 1.387   a: 0.0010    fake_sample: esxvnys_g._j.34u
epoch: 00065  l_G: 1.367   l_D: 1.409   a: 0.0010    fake_sample: x2cgn5_n
epoch: 00066  l_G: 1.378   l_D: 1.395   a: 0.0010    fake_sample: isix6tts._.eiiiloeo
epoch: 00067  l_G: 1.391   l_D: 1.382   a: 0.0010    fake_sample: yo.n_eethmdi102
epoch: 00068  l_G: 1.379   l_D: 1.394   a: 0.0010    fake_sample: mialhoannltri_ks
epoch: 00069  l_G: 1.383   l_D: 1.390   a: 0.0010    fake_sample: inyjeu1y_
epoch: 00070  l_G: 1.383   l_D: 1.390   a: 0.0010    fake_sample: lj_boniyanu9bk
epoch: 00071  l_G: 1.384   l_D: 1.389   a: 0.0010    fake_sample: da2e9i2haisyio_h
epoch: 00072  l_G: 1.390   l_D: 1.383   a: 0.0010    fake_sample: mkuki_emo
epoch: 00073  l_G: 1.387   l_D: 1.386   a: 0.0010    fake_sample: gjairrd
epoch: 00074  l_G: 1.385   l_D: 1.387   a: 0.0010    fake_sample: eeynni_ka3_
epoch: 00075  l_G: 1.339   l_D: 1.439   a: 0.0010    fake_sample: wdned0c
epoch: 00076  l_G: 1.429   l_D: 1.346   a: 0.0010    fake_sample: neo_rnuo2n
epoch: 00077  l_G: 1.451   l_D: 1.324   a: 0.0010    fake_sample: 1ox1nyaneneeoc_oikd
epoch: 00078  l_G: 1.427   l_D: 1.349   a: 0.0010    fake_sample: d_egemeoni_e_
epoch: 00079  l_G: 1.756   l_D: 1.237   a: 0.0010    fake_sample: n
epoch: 00080  l_G: 1.422   l_D: 1.377   a: 0.0010    fake_sample: hdy_ogeuanmesy_5n
epoch: 00081  l_G: 2.731   l_D: 3.385   a: 0.0010    fake_sample: eeamaf2eomoie_miun
epoch: 00082  l_G: 1.304   l_D: 2.744   a: 0.0010    fake_sample: nsojoef
```
