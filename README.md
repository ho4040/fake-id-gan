# Fake id experiment

generate fake id via GANs. 

## info 

cause basic GAN model generate strange words (it was too long), I added some constraint to discriminator like this.

```python
x_hist = tf.contrib.layers.flatten(tf.reduce_sum(x, axis=1))# add histogram of characters
x_len = tf.contrib.layers.flatten(tf.reduce_sum(x, axis=2))# add density of word
layer = tf.concat([layer, x_len, x_hist], 1)
```


# dataset

it requires dataset `real_ids.csv` like this

```
blabla1
blabla__bla1bla1
...
bla23.bla23
```


# requirements

* numpy

* matplotlib

* tensorflow (gpu)


# run

```
$ python3 train.py
```


----

# result example

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

```
