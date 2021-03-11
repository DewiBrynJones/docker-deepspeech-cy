# Sgriptiau Hyfforddi DeepSpeech Mozilla

*Click [here](README_EN.md) to read this page in English*

Mae dogfennaeth gan Mozilla ar DeepSpeech ar gael fan hyn: https://deepspeech.readthedocs.io 

Mae'r sgriptiau canlynol yn cysylltu ag yn hwyluso'r holl gamau a ddilynir er mwyn hyfforddi, gynhyrchu a gwerthuso modelau adnabod lleferydd Cymraeg gyda DeepSpeech Mozilla. Defnyddir setiau Cymraeg o wefan CommonVoice Mozilla fel prif ffynhonnell data hyfforddi. Gydag adnoddau bellach gan Uned Technolegau Iaith, Prifysgol Bangor, mae'r modelau'n addas ar gyfer rhaglenni cynorthwyydd digidol (e.e. Macsen) a thrawsgrifiwr gyffredinol. 

Mae modelau sydd wedi'i hyfforddi'n barod ar gael o'r dudalen cyhoeddi: https://github.com/techiaith/docker-deepspeech-cy/releases


## Rhagofynion

Llwythwch i lawr data lleferydd Cymraeg o wefan CommonVoice: https://voice.mozilla.org/cy/datasets sy'n cael ei ddarparu fel un ffeil mawr wedi'i gwasgu (e.e. `cy.tar.gz`) . Cadwch y ffeil o fewn y ffolder `data/commonvoice`. 

Llwythwch i lawr hefyd Corpws OSCAR o https://oscar-public.huma-num.fr/shuff-orig/cy sy'n cynnwys testunau Cymraeg o'r we. Bydd angen cofrestru er mwyn i'r wefan caniatau lawrlwytho. Cadwch y ffeil o fewn y ffolder `data/oscar`. 


## Paratoi Data

### `import_cv_archive.py`

```shell
root@c67722092f2e:/DeepSpeech# bin/bangor_welsh/import_cv_archive.py --archive /data/commonvoice/cy.tar.gz --target_dir /data/commonvoice/
```

### `analyze_cv.py`

```shell
root@c67722092f2e:/DeepSpeech# /DeepSpeech/bin/bangor_welsh/analyze_cv.py --cv_dir /data/commonvoice/
/data/commonvoice-cy-v5-20200622/clips/dev.csv                0.91 hours      (3269.93 seconds)
/data/commonvoice-cy-v5-20200622/clips/test.csv               0.98 hours      (3514.49 seconds)
/data/commonvoice-cy-v5-20200622/clips/train.csv              1.09 hours      (3941.04 seconds)
/data/commonvoice-cy-v5-20200622/clips/train-all.csv          7.48 hours      (26928.55 seconds)
/data/commonvoice-cy-v5-20200622/clips/other.csv              14.75 hours     (53092.44 seconds)
/data/commonvoice-cy-v5-20200622/clips/validated.csv          58.16 hours     (209380.97 seconds)
```

## Model Acwstig

Defnyddiwch y sgript ganlynol i hyfforddi model acwstig gyda data gan gwefan CommonVoice.

### `run_tl_cv_cy.sh`

Mae'r sgript hon yn defnyddio nodwedd dysgu trosglwyddol (*transfer learning*) DeepSpeech er mwyn cael fudd o ddefnyddio modelau acwstig Saesneg Mozilla, sydd wedi'u hyfforddi ar gasgliadau data llawer mwy o sain, fel man cychwyn ar gyfer hyfforddi adnabod lleferydd Cymraeg.

```shell
root@c67722092f2e:/DeepSpeech# ./bin/bangor_welsh/run_tl_cv_cy.sh --cv_dir /data/commonvoice
```


## Modelau Iaith / Parth Penodol

Nid yw model acwstig ar ei ben ei hun, er ei fod wedi defnyddio technegau dysgu trosglwyddo, yn ddigonol ar gyfer adnabod lleferydd Cymraeg effeithiol. Mae model iaith yn cynorthwyo peiriant adnabod lleferydd i gynhyrchu nid yn unig geiriau dilys (o ddilyniannau ffonemau) ond hefyd ddilyniannau mwy dilys o eiriau.

### `import_bangor_resources.py`

Mae angen rhagor o adnoddau gan Brifysgol Bangor er mwyn hyfforddi DeepSpeech ar gyfer adnabod lleferydd Cymraeg mewn gwahanol gyd-destunau defnyddiol. 

Mae'r sgript isod yn llwytho i lawr rhagor o recordiadau a corpora testun sydd yn galluogi adnabod lleferydd Cymraeg o fewn cynorthwyydd digidol a trawsgrifiwr. Rhaid i chi llwytho i lawr ffeil archif corpws testun OSCAR o flaen llaw er mwyn ei ddefnyddio gyda'r orchymyn isod:

```shell
root@6a88b0d59848:/DeepSpeech# ./bin/bangor_welsh/import_bangor_resources.py --target_dir /data/bangor --oscar_archive /data/oscar/cy.txt.gzip --cv_dir /data/commonvoice/
```

Mae'r sgript mewnforio hefyd yn hidlo unrhyw testunau sy'n anaddas i'r proses hyfforddi modelau iaith adnabod lleferydd ac yn creu copi 'glan' (`.clean`) o'r corpws. 


### `build_lm_scorer.sh`

Dyma'r brif sgript ar gyfer hyfforddi model iaith ac yna ei werthuso gyda model acwstig o gamau blaenorol hyfforddi DeepSpeech. 

##### Ar gyfer defnyddio adnabod lleferydd o fewn Macsen:
```shell
root@6a88b0d59848:/DeepSpeech# ./bin/bangor_welsh/build_lm_scorer.sh --text_file /data/bangor/lm-data/macsen/corpus.clean.txt --domain macsen
```

##### Ar gyfer defnyddio adnabod lleferydd i drawsgrifio:
```shell
root@6a88b0d59848:/DeepSpeech# ./bin/bangor_welsh/build_lm_scorer.sh --text_file /data/bangor/lm-data/oscar/corpus.clean.txt --domain macsen --output_dir /export/macsen --scorer kenlm.scorer
```



### `optimize_lm_scorer.sh`

Bydd y sgript yma yn arbrofi gyda gwahanol baramedrau modelau iaith nes iddo ddod o hyd i'r gwerthoedd gorau posibl sy'n rhoi'r cyfraddau gwallau adnabod lleferydd isaf posibl.
 
Gall y broses gymryd amser hir - oriau neu ddiwrnod neu ddau - gan y bydd yn arbrofi miloedd o weithiau. Yn y diwedd, bydd y sgript yn adrodd ar ddau werth gorau posibl (gelwir yn 'alpha' a 'beta') ac yn gofyn ichi eu mewnbynnu er mwyn eu cynnwys ym mhecyn terfynol y model iaith.

```shell
root@6a88b0d59848:/DeepSpeech# ./bin/bangor_welsh/optimize_lm_scorer.sh --csv_test_file /data/bangor/testsets/data/macsen/deepspeech.csv --domain macsen [--checkpoint_dir /checkpoints/cy]
```


## Profi'r modelau

Er mwyn gwybod pa mor dda neu ddim mae'r modelau, mae modd profi erbyn set profi sydd wedi ei fanylu o fewn ffeil CSV. Er enghraifft, er mwyn profi'r modelau trawsgrifio:

```shell
root@6a88b0d59848:/DeepSpeech# ./bin/bangor_welsh/evaluate.sh --csv_test_file /data/bangor/testsets/data/trawsgrifio/arddweud_200617/deepspeech.csv --scorer /export/transcribe/kenlm.transcribe.scorer
```

