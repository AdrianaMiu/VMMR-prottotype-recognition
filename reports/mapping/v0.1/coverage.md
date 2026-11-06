# Generation mapping coverage report v0.1.0

- mapping version: 0.1.0
- queue file: `generation_mapping_queue_v0.1.csv`
- sha256: 390336d5a1be33959302042b7c839e4359ddb2fbdeb7a9f402e67899fd303c17  generation_mapping_queue_v0.1.csv

Every queue row is `provisional` and carries only what the CompCars folder structure states (ids, names, folder year). No generation interval, identifier, facelift phase, source or reviewer has been filled in, by hand or by any automated tool: those fields stay empty until the named human reviewers approve them (PROTOCOL.md §6 rules 8 and 10).

## Model-year records

| total | mapped | unmapped |
|---|---|---|
| 4428 | 0 | 4428 |

## Images

| total | retained | excluded |
|---|---|---|
| 136726 | 136365 | 361 |

| exclusion reason | images |
|---|---|
| degenerate_bbox | 10 |
| invalid_year_folder | 8 |
| unknown_year | 343 |

## Mapping status distribution

| mapping_status | rows |
|---|---|
| provisional | 4428 |

## Candidate class counts

Makes: 163. Models with at least one valid model-year: 1710; models in the manifest whose every image is excluded (no valid year): 6.

| hypothesis | classes |
|---|---|
| every CompCars year is its own generation | 4428 |
| all years of a model form one generation | 1710 |

The approved mapping will land between these two bounds.

## Models whose year list suggests more than one generation

Flag `span`: inclusive span of recorded years >= 7. Flag `gap`: a gap >= 3 years between consecutive recorded years. 353 models flagged; these are review hints, not labels.

| make | model | model_id | years | span | max gap | flags |
|---|---|---|---|---|---|---|
| Dacia | Duster | 1040 | 2009;2011;2014 | 6 | 3 | gap |
| Dacia | Logan | 1041 | 2005;2006;2007;2013;2014 | 10 | 6 | span;gap |
| Dacia | Sandero | 1042 | 2009;2010;2013 | 5 | 3 | gap |
| GMC | Yukon | 682 | 2007;2008;2013;2015 | 9 | 5 | span;gap |
| GMC | Acadia | 685 | 2007;2011;2013 | 7 | 4 | span;gap |
| GMC | Savana | 687 | 2006;2010;2011;2012;2013;2014 | 9 | 4 | span;gap |
| Jeep | Free | 739 | 2004;2008;2010;2012 | 9 | 4 | span;gap |
| Jeep | Grand Cherokee | 745 | 2004;2005;2010;2011;2012;2013;2014 | 11 | 5 | span;gap |
| KTM | X-BOW | 1318 | 2008;2014 | 7 | 6 | span;gap |
| MINI | MINI | 827 | 2009;2011;2012;2013;2014;2015 | 7 | 2 | span |
| MINI | MINI CLUBMAN | 828 | 2008;2009;2011;2013;2014 | 7 | 2 | span |
| MINI | MINI COUPE | 830 | 2009;2012 | 4 | 3 | gap |
| Noble | Noble M600 | 2001 | 2010;2015 | 6 | 5 | gap |
| PGO | CEVENNES | 1048 | 2009;2013 | 5 | 4 | gap |
| Scion | Scion xB | 1304 | 2008;2012;2014 | 7 | 4 | span;gap |
| TESLA | MODEL S | 948 | 2009;2014;2015 | 7 | 5 | span;gap |
| Zenvo | Zenvo ST1 | 1757 | 2010;2015 | 6 | 5 | gap |
| Yiqi | Weizi | 1069 | 2004;2009;2011 | 8 | 5 | span;gap |
| Yiqi | Xiali sedan | 1073 | 2008;2012 | 5 | 4 | gap |
| Mitsubishi | Pajero | 1866 | 2008;2010;2011;2012;2014;2015 | 8 | 2 | span |
| Mitsubishi | Grandis | 1867 | 2004;2006;2007;2008;2009;2010 | 7 | 2 | span |
| Mitsubishi | Evo | 1871 | 2006;2008;2012 | 7 | 4 | span;gap |
| Mitsubishi | Mitsubishi i | 1872 | 2008;2011;2012 | 5 | 3 | gap |
| Mitsubishi | Mitsubishi Colt | 1873 | 2005;2009 | 5 | 4 | gap |
| Mitsubishi | Endeavor | 1878 | 2004;2006;2007;2011 | 8 | 4 | span;gap |
| Mitsubishi | Mitsubishi Lancer | 1888 | 2006;2007;2008;2010;2012;2015 | 10 | 3 | span;gap |
| Mitsubishi | Mitsubishi Galant | 1891 | 2007;2009;2012 | 6 | 3 | gap |
| Dongnan | Delica | 535 | 2008;2010;2014 | 7 | 4 | span;gap |
| Dongfengxiaokang | DFSK K07 | 573 | 2006;2008;2010;2012;2013 | 8 | 2 | span |
| Dongfengxiaokang | DFSK K07II | 574 | 2007;2011;2013;2014 | 8 | 4 | span;gap |
| Dongfengfengdu | Ruiqi Pickup | 565 | 2009;2010;2013;2015 | 7 | 3 | span;gap |
| Dongfengfengxing | Jingyi | 554 | 2007;2009;2010;2011;2012;2014 | 8 | 2 | span |
| Zhonghua | Zhonghua Zunchi | 1959 | 2004;2007;2008;2009;2011 | 8 | 3 | span;gap |
| Toyota | Coaster | 1324 | 2007;2013 | 7 | 6 | span;gap |
| Toyota | Lcruiser | 1325 | 2007;2010;2012;2013 | 7 | 3 | span;gap |
| Toyota | Prado | 1326 | 2006;2010;2014 | 9 | 4 | span;gap |
| Toyota | Prius | 1327 | 2005;2006;2012 | 8 | 6 | span;gap |
| Toyota | FJ Cruiser | 1331 | 2007;2010;2011;2013;2014 | 8 | 3 | span;gap |
| Toyota | Venza | 1332 | 2009;2013 | 5 | 4 | gap |
| Toyota | Toyota 4 Rinner | 1334 | 2010;2011;2014 | 5 | 3 | gap |
| Toyota | Previa | 1341 | 2004;2006;2007;2008;2010;2012 | 9 | 2 | span |
| Toyota | Saina | 1342 | 2009;2011;2012;2015 | 7 | 3 | span;gap |
| Toyota | Toyota IQ | 1343 | 2009;2013 | 5 | 4 | gap |
| Toyota | Toyota Aygo | 1344 | 2009;2013;2014;2015 | 7 | 4 | span;gap |
| Toyota | Avensis | 1345 | 2009;2012;2015 | 7 | 3 | span;gap |
| Toyota | Undra | 1349 | 2009;2011;2014;2015 | 7 | 3 | span;gap |
| Toyota | Toyota Auris | 1350 | 2010;2011;2013;2016 | 7 | 3 | span;gap |
| Toyota | 丰田Avalon | 1370 | 2006;2011;2013;2015 | 10 | 5 | span;gap |
| Toyota | Camry | 1386 | 2008;2009;2010;2011;2012;2013;2015 | 8 | 2 | span |
| Toyota | Yaris | 1389 | 2008;2009;2011;2013;2015 | 8 | 2 | span |
| Wuling | Wulinghongguang | 1899 | 2010;2013;2014 | 5 | 3 | gap |
| Wuling | Wulingzhiguang | 1901 | 2010;2013;2014 | 5 | 3 | gap |
| Wuling | Wulingrongguang | 1904 | 2008;2011;2012;2014 | 7 | 3 | span;gap |
| AC Chnitzer | AC Schnitzer X5 | 1770 | 2008;2010;2011;2014 | 7 | 3 | span;gap |
| AC Chnitzer | AC Schnitzer X6 | 1771 | 2008;2010;2011;2014;2015 | 8 | 3 | span;gap |
| Zoyte | Zotye T600 | 1030 | 2011;2014 | 4 | 3 | gap |
| Iveco | Power Daily | 1288 | 2009;2012;2013;2014 | 6 | 3 | gap |
| Iveco | Turbo Daily | 1293 | 2008;2011;2013 | 6 | 3 | gap |
| Porsche | Boxster | 315 | 2009;2013;2014 | 6 | 4 | gap |
| Porsche | Panamera | 317 | 2009;2010;2012;2013;2014;2015 | 7 | 2 | span |
| Porsche | Canyenne hybrid | 322 | 2011;2015 | 5 | 4 | gap |
| Porsche | Cayman | 326 | 2009;2012;2013;2014;2016 | 8 | 3 | span;gap |
| Mitsuoka | Orochi | 1435 | 2010;2011;2014 | 5 | 3 | gap |
| Chrysler | Prowler | 1123 | 2002;2011 | 10 | 9 | span;gap |
| Chrysler | Chrysler 300C | 1131 | 2006;2012;2013;2014 | 9 | 6 | span;gap |
| Chrysler | Grand Voager | 1132 | 2007;2012;2013;2014 | 8 | 5 | span;gap |
| Lamorghini | Gallardo | 1159 | 2008;2009;2010;2011;2012;2013;2014 | 7 | 1 | span |
| Cadillac | Escalade | 771 | 2010;2013;2014;2015 | 6 | 3 | gap |
| Buck | Buick GL8 Business | 189 | 2008;2011;2012;2013;2014 | 7 | 3 | span;gap |
| Buck | Lacrosse | 192 | 2008;2009;2010;2011;2012;2013;2014 | 7 | 1 | span |
| Buck | Buick Riviera | 200 | 1965;2013 | 49 | 48 | span;gap |
| Buck | Enclave | 202 | 2009;2010;2013;2014 | 6 | 3 | gap |
| Lifan | Lifan 520 | 1590 | 2006;2007;2008;2010;2011;2012 | 7 | 2 | span |
| BAW | BAW BJ40 | 327 | 2010;2014 | 5 | 4 | gap |
| Baihc | Luba | 356 | 2006;2012 | 7 | 6 | span;gap |
| Baihc | Yongshi | 360 | 2008;2009;2014 | 7 | 5 | span;gap |
| Huatai | Tekala | 1454 | 2004;2005;2008;2009;2010;2012 | 9 | 3 | span;gap |
| Shuanghuan | Shuanghuan SCEO | 1698 | 2005;2006;2009;2010;2011 | 7 | 3 | span;gap |
| Shuanglong | Actyon | 933 | 2008;2010;2011;2013;2014 | 7 | 2 | span |
| Shuanglong | Rexton | 934 | 2004;2006;2011;2013;2014 | 11 | 5 | span;gap |
| Shuanglong | Rodius | 936 | 2007;2008;2013;2014 | 8 | 5 | span;gap |
| Shuanglong | Kyron | 937 | 2006;2007;2010;2011 | 6 | 3 | gap |
| Shuanglong | Chairman | 938 | 2002;2004;2011;2012 | 11 | 7 | span;gap |
| Geely | Geely TX4 | 1480 | 2009;2012 | 4 | 3 | gap |
| Geely | King Kong CROSS | 1481 | 2011;2014 | 4 | 3 | gap |
| Geely | King Kong hatchback | 1482 | 2011;2014 | 4 | 3 | gap |
| Geely | King Kong sedan | 1483 | 2007;2009;2010;2011;2013;2014 | 8 | 2 | span |
| Geely | Panda CROSS | 1491 | 2011;2014 | 4 | 3 | gap |
| Geely | Vision | 1492 | 2006;2009;2010;2011;2013;2014;2015 | 10 | 3 | span;gap |
| Geely | Ziyoujian | 1494 | 2006;2010;2011;2012;2015 | 10 | 4 | span;gap |
| Hafei | Saima | 712 | 2004;2007;2009 | 6 | 3 | gap |
| Hafei | Zhongyi V5 | 714 | 2009;2013 | 5 | 4 | gap |
| Hafei | Lubao | 716 | 2004;2008;2009;2010;2011 | 8 | 4 | span;gap |
| Volkswagen | Cross Polo | 434 | 2008;2012;2014 | 7 | 4 | span;gap |
| Volkswagen | GOL | 437 | 2004;2012;2013 | 10 | 8 | span;gap |
| Volkswagen | Lavide | 440 | 2008;2011;2013;2014;2015 | 8 | 3 | span;gap |
| Volkswagen | Passat Lingyu | 442 | 2005;2007;2009;2012 | 8 | 3 | span;gap |
| Volkswagen | Santana Vista | 444 | 2004;2006;2008;2010 | 7 | 2 | span |
| Volkswagen | Touran | 445 | 2008;2011;2013;2015 | 8 | 3 | span;gap |
| Volkswagen | Volkswagen Eos | 453 | 2008;2010;2011;2014;2015 | 8 | 3 | span;gap |
| Volkswagen | Golf R | 456 | 2010;2011;2014;2015 | 6 | 3 | gap |
| Volkswagen | Multivan | 463 | 2006;2008;2012;2014;2015 | 10 | 4 | span;gap |
| Volkswagen | Scirocco | 465 | 2009;2010;2011;2013;2015 | 7 | 2 | span |
| Volkswagen | Scirocco R | 466 | 2011;2014 | 4 | 3 | gap |
| Volkswagen | Touareg | 467 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 | span |
| Volkswagen | Polo abroad version | 470 | 2009;2011;2013;2014;2015 | 7 | 2 | span |
| Volkswagen | Beetle convertible | 472 | 2009;2010;2015 | 7 | 5 | span;gap |
| Volkswagen | Cross Coupe | 475 | 2012;2015 | 4 | 3 | gap |
| Volkswagen | Volkswagen XL1 | 479 | 2009;2011;2014 | 6 | 3 | gap |
| Volkswagen | Volkswagen up | 496 | 2009;2013;2014;2015 | 7 | 4 | span;gap |
| Volkswagen | Volkswagen Fox | 497 | 2004;2015 | 12 | 11 | span;gap |
| Volkswagen | Caddy abroad version | 498 | 2011;2013;2016 | 6 | 3 | gap |
| Volkswagen | Bora | 500 | 2008;2011;2012;2013;2014 | 7 | 3 | span;gap |
| Volkswagen | Golf GTI | 502 | 2009;2010;2012;2013;2015 | 7 | 2 | span |
| Volkswagen | Jetta | 503 | 2009;2010;2011;2012;2013;2015 | 7 | 2 | span |
| Volkswagen | Magotan | 504 | 2009;2011;2012;2013;2015 | 7 | 2 | span |
| Chrey | Eastar | 1651 | 2004;2005;2006;2007;2012 | 9 | 5 | span;gap |
| Benz | Benz GLK Class | 126 | 2008;2010;2011;2012;2013;2014 | 7 | 2 | span |
| Benz | Benz E Class | 127 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 | span |
| Benz | Benz CLS Class AMG | 129 | 2009;2012;2014;2015 | 7 | 3 | span;gap |
| Benz | Benz C Class AMG | 130 | 2009;2010;2012;2014;2015 | 7 | 2 | span |
| Benz | Benz G Class AMG | 132 | 2009;2010;2013;2014 | 6 | 3 | gap |
| Benz | Benz M Class AMG | 133 | 2007;2011;2012;2014;2015 | 9 | 4 | span;gap |
| Benz | Benz SL Class AMG | 136 | 2009;2013;2014 | 6 | 4 | gap |
| Benz | Benz S Class AMG | 137 | 2010;2013;2014;2015 | 6 | 3 | gap |
| Benz | Benz C Class estate | 152 | 2010;2011;2014 | 5 | 3 | gap |
| Benz | Benz G Class | 157 | 2010;2013;2014;2015 | 6 | 3 | gap |
| Benz | Benz R Class | 159 | 2006;2010;2011;2014 | 9 | 4 | span;gap |
| Benz | Benz SLK Class | 160 | 2009;2010;2011;2012;2015 | 7 | 3 | span;gap |
| Benz | Benz S Class hybird | 163 | 2009;2010;2014 | 6 | 4 | gap |
| Benz | Sprinter abroad version | 164 | 2009;2011;2013;2014;2015 | 7 | 2 | span |
| Benz | Unimog | 185 | 2000;2010;2011;2012 | 13 | 10 | span;gap |
| Audi | Audi A4L | 2 | 2009;2010;2011;2012;2013;2015 | 7 | 2 | span |
| Audi | Audi A6 | 9 | 2004;2012;2013;2014;2015 | 12 | 8 | span;gap |
| Audi | Audi RS4 | 14 | 2006;2013;2014 | 9 | 7 | span;gap |
| Audi | Audi RS3 | 16 | 2012;2015;2016 | 5 | 3 | gap |
| Audi | Audi RS6 | 18 | 2002;2009;2013;2015 | 14 | 7 | span;gap |
| Audi | Audi A5 coupe | 23 | 2008;2009;2010;2012;2013;2014 | 7 | 2 | span |
| Audi | Audi A7 | 26 | 2009;2012;2013;2014;2015 | 7 | 3 | span;gap |
| Audi | Audi A8L | 27 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 | span |
| Audi | Audi R8 | 33 | 2010;2011;2013;2014;2015;2016 | 7 | 2 | span |
| Audi | Audi S5 coupe | 35 | 2008;2010;2012;2014 | 7 | 2 | span |
| Audi | Audi S8 | 39 | 2008;2009;2013;2014;2015 | 8 | 4 | span;gap |
| Audi | Audi A2 | 46 | 2002;2011;2013 | 12 | 9 | span;gap |
| Audi | Audi quattro | 47 | 2010;2013;2014 | 5 | 3 | gap |
| Audi | Audi A3 e-tron | 56 | 2011;2015 | 5 | 4 | gap |
| Audi | Audi S4 | 63 | 2008;2013 | 6 | 5 | gap |
| Wisemann | Wiesmann Roadster | 1725 | 2006;2012 | 7 | 6 | span;gap |
| Wisemann | Wiesmann GT | 1727 | 2006;2009;2012;2013 | 8 | 3 | span;gap |
| BWM | BWM 3 Series | 68 | 2008;2009;2010;2011;2012;2013;2014 | 7 | 1 | span |
| BWM | BWM M3 | 73 | 2008;2014 | 7 | 6 | span;gap |
| BWM | BWM M5 | 75 | 2005;2011;2012;2013;2014 | 10 | 6 | span;gap |
| BWM | BWM M6 coupe | 76 | 2008;2013;2014 | 7 | 5 | span;gap |
| BWM | BWM X5 M | 78 | 2010;2013;2014;2016 | 7 | 3 | span;gap |
| BWM | BWM X6 M | 79 | 2010;2013;2014;2015;2016 | 7 | 3 | span;gap |
| BWM | BWM 6 Series convertible | 98 | 2008;2011 | 4 | 3 | gap |
| BWM | BWM 7 Series | 101 | 2009;2010;2011;2013;2014;2015 | 7 | 2 | span |
| BWM | BWM X5 | 105 | 2008;2009;2011;2012;2013;2014 | 7 | 2 | span |
| Bentley | Arnage | 373 | 2006;2009 | 4 | 3 | gap |
| Brabus | Brabus smart fortwo | 1965 | 2005;2012;2013;2014;2015 | 11 | 7 | span;gap |
| Brabus | Brabus SL Class | 1973 | 2000;2010;2013 | 14 | 10 | span;gap |
| Bugatti | Veyron | 375 | 2008;2011;2012;2013;2014;2015 | 8 | 3 | span;gap |
| Pagani | Zonda | 1314 | 2004;2005;2006;2008;2011;2013 | 10 | 3 | span;gap |
| Guangqichuanqi | E-jet | 676 | 2010;2014 | 5 | 4 | gap |
| GAC | Caiyun 100 | 1418 | 2009;2012 | 4 | 3 | gap |
| Karry | Youya | 1785 | 2007;2010;2012;2013 | 7 | 3 | span;gap |
| Jaguar | Gaguar XK | 748 | 2006;2009;2012;2013;2014 | 9 | 3 | span;gap |
| Jaguar | Gaguar C-X75 | 749 | 2010;2013 | 4 | 3 | gap |
| Jaguar | Gaguar C-TYPE | 752 | 1951;2011 | 61 | 60 | span;gap |
| Jaguar | Gaguar E-TYPE | 753 | 1971;2011;2014 | 44 | 40 | span;gap |
| Jaguar | Gaguar XF | 756 | 2009;2011;2012;2013;2014;2015 | 7 | 2 | span |
| Subaru | Outback | 1230 | 2008;2010;2011;2012;2013;2014;2015 | 8 | 2 | span |
| Subaru | Legacy | 1231 | 2009;2010;2013;2014;2015 | 7 | 3 | span;gap |
| Subaru | Tribeca | 1243 | 2005;2007;2008;2011;2012 | 8 | 3 | span;gap |
| Skoda | Fabia | 910 | 2008;2011;2012;2013;2015 | 8 | 3 | span;gap |
| Skoda | Octavia | 911 | 2009;2010;2012;2013;2014;2015 | 7 | 2 | span |
| Skoda | Octavia RS | 912 | 2010;2011;2014 | 5 | 3 | gap |
| Skoda | Yeti abroad version | 919 | 2010;2013;2014;2015 | 6 | 3 | gap |
| Skoda | Roomster | 923 | 2006;2007;2011 | 6 | 4 | gap |
| Nissan | X-Trail | 853 | 2008;2010;2012;2014 | 7 | 2 | span |
| Nissan | Teana | 854 | 2008;2010;2011;2012;2013;2014 | 7 | 2 | span |
| Nissan | Sylphy | 856 | 2006;2009;2010;2012 | 7 | 3 | span;gap |
| Nissan | Sunshine | 857 | 2011;2014 | 4 | 3 | gap |
| Nissan | Livina | 860 | 2010;2013 | 4 | 3 | gap |
| Nissan | Ma Chi | 864 | 2010;2011;2014;2015 | 6 | 3 | gap |
| Nissan | Paladin | 865 | 2004;2006;2013 | 10 | 7 | span;gap |
| Nissan | Nissan D22 Cargo | 867 | 2008;2011 | 4 | 3 | gap |
| Nissan | Nissan 370Z | 871 | 2009;2013;2014;2015 | 7 | 4 | span;gap |
| Nissan | Nissan GT-R | 872 | 2009;2010;2012;2013;2014;2015 | 7 | 2 | span |
| Nissan | Patrol | 874 | 2004;2012;2013;2014 | 11 | 8 | span;gap |
| Nissan | Nissan Cube | 875 | 2009;2013 | 5 | 4 | gap |
| Nissan | Pathfinder | 879 | 2005;2010;2012;2013 | 9 | 5 | span;gap |
| Nissan | Rogue | 899 | 2011;2014 | 4 | 3 | gap |
| Nissan | Quest | 905 | 2005;2006;2011;2012;2013;2015 | 11 | 5 | span;gap |
| Nissan | Altima | 909 | 2005;2010;2012;2013;2015 | 11 | 5 | span;gap |
| Changhe | Ideal | 431 | 2004;2006;2011 | 8 | 5 | span;gap |
| Honda | Odyssey | 209 | 2009;2011;2013;2014;2015 | 7 | 2 | span |
| Honda | Fit | 215 | 2008;2011;2013;2014 | 7 | 3 | span;gap |
| Honda | Honda NSX | 226 | 2002;2003;2012;2015 | 14 | 9 | span;gap |
| Honda | Spirior | 250 | 2009;2013;2014;2015 | 7 | 4 | span;gap |
| Honda | Civic | 251 | 2009;2012;2013;2014;2015 | 7 | 3 | span;gap |
| Lincoln | Lincoln MKX | 818 | 2009;2010;2011;2013;2014;2016 | 8 | 2 | span |
| Lincoln | Lincoln MKZ | 819 | 2007;2010;2011;2013;2014 | 8 | 3 | span;gap |
| Peugeot | CROSS 307 | 265 | 2004;2006;2007;2008;2012;2013 | 10 | 4 | span;gap |
| Peugeot | Peugeot 207CC | 266 | 2008;2011;2012;2013 | 6 | 3 | gap |
| Peugeot | Peugeot 107 | 271 | 2009;2013 | 5 | 4 | gap |
| Peugeot | Peugeot 5008 | 282 | 2010;2014 | 5 | 4 | gap |
| Opel | GTC hatchback | 1845 | 2004;2008;2010;2012;2013;2014 | 11 | 4 | span;gap |
| Opel | Opel CORSA | 1849 | 2008;2011;2015 | 8 | 4 | span;gap |
| Opel | Meriva | 1856 | 2006;2013;2014 | 9 | 7 | span;gap |
| BYD | BYD F0 | 294 | 2009;2010;2011;2012;2015 | 7 | 3 | span;gap |
| BYD | BYD e6 | 306 | 2009;2012;2014 | 6 | 3 | gap |
| BYD | BYD F3 | 314 | 2009;2011;2012;2013;2014;2015 | 7 | 2 | span |
| Jonway | Yongyuan A380 | 1313 | 2006;2008;2010;2012;2013;2014 | 9 | 2 | span |
| Jianghuai | Ruifeng | 1524 | 2007;2011;2012 | 6 | 4 | gap |
| Jianghuai | Tongyue RS | 1533 | 2009;2012 | 4 | 3 | gap |
| Jiangling | Baodian | 1544 | 2007;2012;2013 | 7 | 5 | span;gap |
| Vauxhall | Vauxhall Antara | 1092 | 2007;2011 | 5 | 4 | gap |
| Vauxhall | Vauxhall Corsa | 1095 | 2011;2015 | 5 | 4 | gap |
| Vauxhall | Vauxhall VXR8 | 1099 | 2007;2009;2011;2013;2014 | 8 | 2 | span |
| Vauxhall | Vauxhall Astra | 1100 | 2005;2006;2013 | 9 | 7 | span;gap |
| Volvo | Volvo V40 | 1703 | 2009;2013;2014;2015 | 7 | 4 | span;gap |
| Volvo | Volvo XC60 | 1706 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 | span |
| Volvo | Volvo XC90 abroad version | 1707 | 2008;2010;2011;2012;2013;2015 | 8 | 2 | span |
| Volvo | Volvo S80 | 1708 | 2006;2008;2012;2013;2014 | 9 | 4 | span;gap |
| Volvo | Volvo V70 | 1710 | 2009;2012;2014 | 6 | 3 | gap |
| Volvo | Volvo XC70 | 1711 | 2004;2007;2008;2011;2012;2014 | 11 | 3 | span;gap |
| Volvo | Universe | 1713 | 2011;2015 | 5 | 4 | gap |
| Ferrari | California | 662 | 2009;2012;2015 | 7 | 3 | span;gap |
| Haima | Fumeilai | 1437 | 2008;2010;2011;2012;2013;2014 | 7 | 2 | span |
| Haima | Premacy | 1447 | 2006;2009;2010;2012;2013;2014 | 9 | 3 | span;gap |
| Haima(Zhengzhou) | FOS | 700 | 2009;2012 | 4 | 3 | gap |
| Cheetah | Feiteng | 1154 | 2007;2008;2009;2010;2013 | 7 | 3 | span;gap |
| Maserati | Quattroporte | 1169 | 2009;2011;2013;2014;2015 | 7 | 2 | span |
| Maserati | Maserati GT | 1174 | 2008;2009;2011;2012;2013;2014;2015 | 8 | 2 | span |
| Hyundai | i30 | 955 | 2007;2009;2013 | 7 | 4 | span;gap |
| Hyundai | Rena | 962 | 2010;2014 | 5 | 4 | gap |
| Hyundai | Verna | 963 | 2011;2014 | 4 | 3 | gap |
| Hyundai | Tucson | 965 | 2006;2009;2013 | 8 | 4 | span;gap |
| Hyundai | Accent | 966 | 2006;2008;2011 | 6 | 3 | gap |
| Hyundai | Elantra | 967 | 2004;2005;2006;2007;2011 | 8 | 4 | span;gap |
| Hyundai | Rohens Coupe | 977 | 2009;2012 | 4 | 3 | gap |
| Hyundai | i20 | 982 | 2009;2013;2015 | 7 | 4 | span;gap |
| Hyundai | Tucson abroad vresion | 983 | 2005;2010;2013;2014;2015 | 11 | 5 | span;gap |
| Hyundai | Sonata abroad version | 1000 | 2011;2014;2015 | 5 | 3 | gap |
| Ford | Classic Focus hatchback | 587 | 2008;2011;2012;2013 | 6 | 3 | gap |
| Ford | Classic Focus sedan | 588 | 2008;2011;2012;2013 | 6 | 3 | gap |
| Ford | S-MAX | 589 | 2007;2008;2011;2013;2014;2015 | 9 | 3 | span;gap |
| Ford | New Focus hatchback | 590 | 2006;2007;2009;2011;2012;2013 | 8 | 2 | span |
| Ford | New Focus sedan | 591 | 2005;2009;2011;2012;2013 | 9 | 4 | span;gap |
| Ford | Kuga | 594 | 2008;2010;2013;2014;2015 | 8 | 3 | span;gap |
| Ford | Transit | 599 | 2009;2010;2013;2014 | 6 | 3 | gap |
| Ford | Ford F-250 | 602 | 2006;2012 | 7 | 6 | span;gap |
| Ford | Explorer | 607 | 2011;2013;2016 | 6 | 3 | gap |
| Ford | Ford Ka | 609 | 2008;2011 | 4 | 3 | gap |
| Ford | Ford Flex | 610 | 2004;2009;2013 | 10 | 5 | span;gap |
| Ford | Ford Taurus | 612 | 2009;2013 | 5 | 4 | gap |
| Ford | Ford Fusion | 613 | 2010;2013 | 4 | 3 | gap |
| Ford | Falcon | 622 | 2008;2014 | 7 | 6 | span;gap |
| Ford | Ranger | 624 | 2008;2012 | 5 | 4 | gap |
| Ford | Taurus | 640 | 2010;2013;2015 | 6 | 3 | gap |
| Futian | Midi | 1400 | 2009;2010;2011;2014 | 6 | 3 | gap |
| Futian | Fukuda | 1401 | 2011;2015 | 5 | 4 | gap |
| Infiniti | Infiniti Q60 | 1008 | 2010;2013;2014;2015 | 6 | 3 | gap |
| Infiniti | Infiniti Q60S | 1009 | 2009;2013 | 5 | 4 | gap |
| Infiniti | Infiniti QX50 | 1012 | 2008;2009;2010;2011;2013;2014;2015 | 8 | 2 | span |
| Infiniti | Infiniti QX70 | 1015 | 2009;2010;2011;2013;2014;2015 | 7 | 2 | span |
| Lotus | Lotus L3 hatchback | 1603 | 2009;2013 | 5 | 4 | gap |
| Lotus | Lotus L3 sedan | 1605 | 2009;2010;2013 | 5 | 3 | gap |
| FIAT | Palio | 643 | 2004;2005;2006;2012 | 9 | 6 | span;gap |
| FIAT | Punto | 649 | 2008;2011;2012 | 5 | 3 | gap |
| FIAT | Doblo | 651 | 2005;2010;2015 | 11 | 5 | span;gap |
| FIAT | FIAT Idea | 654 | 2011;2014 | 4 | 3 | gap |
| FIAT | FIAT Uno | 657 | 2011;2015 | 5 | 4 | gap |
| FIAT | FIAT 500 | 661 | 2009;2011;2012;2013;2014;2015 | 7 | 2 | span |
| Lancia | Lancia Ypsilon | 1781 | 2006;2012 | 7 | 6 | span;gap |
| Acura | Acura MDX | 1640 | 2008;2010;2011;2013;2014 | 7 | 2 | span |
| Acura | Acura ZDX | 1645 | 2009;2012;2013 | 5 | 3 | gap |
| Acura | Acura NSX | 1646 | 2012;2015;2016 | 5 | 3 | gap |
| KIA | KIA K2 sedan | 1181 | 2011;2012;2015 | 5 | 3 | gap |
| KIA | RIO | 1186 | 2005;2007;2012 | 8 | 5 | span;gap |
| KIA | Cerato | 1187 | 2005;2006;2007;2008;2010;2012 | 8 | 2 | span |
| KIA | Borrego | 1195 | 2008;2012;2013;2015 | 8 | 4 | span;gap |
| KIA | KIA VQ | 1197 | 2006;2008;2011;2012;2013 | 8 | 3 | span;gap |
| KIA | Sorento | 1200 | 2009;2010;2012;2013;2014;2015 | 7 | 2 | span |
| KIA | KIA venga | 1202 | 2010;2011;2015 | 6 | 4 | gap |
| KIA | SPORTAGE | 1203 | 2005;2011;2014 | 10 | 6 | span;gap |
| KIA | KIA Picanto | 1205 | 2012;2015 | 4 | 3 | gap |
| KIA | KIA Ceed | 1206 | 2010;2013;2014;2015 | 6 | 3 | gap |
| KIA | KIA Trackster | 1207 | 2012;2015 | 4 | 3 | gap |
| KIA | KIA Forte | 1210 | 2010;2011;2014 | 5 | 3 | gap |
| Lotus | Elan | 1822 | 1962;2010 | 49 | 48 | span;gap |
| LAND-ROVER | Defender | 1141 | 2004;2013;2014;2015 | 12 | 9 | span;gap |
| Maybach | Maybach | 1638 | 2007;2010;2011 | 5 | 3 | gap |
| Dodge | Challenger | 519 | 2011;2014;2015 | 5 | 3 | gap |
| Dodge | Durango | 520 | 2011;2014 | 4 | 3 | gap |
| Dodge | Avenger | 521 | 2007;2008;2011;2013;2014 | 8 | 3 | span;gap |
| Dodge | Rampage | 526 | 2006;2011;2013;2014;2015 | 10 | 5 | span;gap |
| Dodge | Challenger SRT | 530 | 2012;2015 | 4 | 3 | gap |
| Mustang | Mustang F16 | 1739 | 2011;2014 | 4 | 3 | gap |
| Mustang | Mustang T70 | 1742 | 2012;2015 | 4 | 3 | gap |
| Jinbei | Geruisi | 1555 | 2008;2009;2010;2013;2014 | 7 | 3 | span;gap |
| Jinbei | Haishi | 1556 | 2008;2011;2012;2014 | 7 | 3 | span;gap |
| Jinbei | Jindian | 1559 | 2009;2014 | 6 | 5 | gap |
| Jinbei | Leilong | 1560 | 2009;2014 | 6 | 5 | gap |
| Jinbei | Dalishen | 1564 | 2009;2013;2014 | 6 | 4 | gap |
| Suzuki | Cultus | 1791 | 2004;2007;2009;2011;2012;2013 | 10 | 3 | span;gap |
| Suzuki | Grand Vitara | 1798 | 2008;2009;2011;2012;2013;2015 | 8 | 2 | span |
| Suzuki | Jimny | 1811 | 2011;2012;2015 | 5 | 3 | gap |
| Suzuki | Liana hatchback | 1815 | 2008;2011;2012 | 5 | 3 | gap |
| GreatWall | Wingle 3 | 383 | 2007;2011 | 5 | 4 | gap |
| GreatWall | Oula | 393 | 2008;2013 | 6 | 5 | gap |
| Changan Business | Changan Xingka | 423 | 2009;2013 | 5 | 4 | gap |
| Changan Business | Changan Star | 424 | 2009;2012;2013 | 5 | 3 | gap |
| Changan Business | Changan Star 7 | 425 | 2010;2011;2014 | 5 | 3 | gap |
| Changan | Changan CX20 | 401 | 2011;2014 | 4 | 3 | gap |
| Alfa Romeo | MiTo | 1746 | 2009;2010;2014;2015 | 7 | 4 | span;gap |
| Aston Martin | DB9 | 1051 | 2007;2011;2013;2014 | 8 | 4 | span;gap |
| Aston Martin | Rapide | 1052 | 2010;2013;2014;2015 | 6 | 3 | gap |
| Aston Martin | V12 Vantage | 1053 | 2009;2012;2014;2015 | 7 | 3 | span;gap |
| Aston Martin | ONE-77 | 1058 | 2010;2013 | 4 | 3 | gap |
| Aston Martin | Lagonda | 1060 | 2009;2013;2015;2016 | 8 | 4 | span;gap |
| Lufeng | Landwind X8 | 811 | 2009;2011;2012;2014;2015 | 7 | 2 | span |
| Shanqitongjia | Fujia | 1700 | 2011;2014 | 4 | 3 | gap |
| Chevy | AVEO hatchback | 1905 | 2011;2014 | 4 | 3 | gap |
| Chevy | Epica | 1907 | 2006;2007;2010;2011;2012;2013 | 8 | 3 | span;gap |
| Chevy | Cruze sedan | 1908 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 | span |
| Chevy | CHEVY Traverse | 1921 | 2009;2013 | 5 | 4 | gap |
| Chevy | CHEVY Orlando | 1924 | 2009;2012 | 4 | 3 | gap |
| Chevy | Equinox | 1934 | 2005;2008;2010;2013;2015;2016 | 12 | 3 | span;gap |
| Chevy | CHEVY Tahoe | 1941 | 2002;2007;2015 | 14 | 8 | span;gap |
| Chevy | Suburban | 1944 | 2007;2015 | 9 | 8 | span;gap |
| Citroen | Citroen C4 | 1266 | 2006;2009;2011;2013;2015 | 10 | 3 | span;gap |
| Citroen | Citroen C3 | 1267 | 2004;2008;2010;2014 | 11 | 4 | span;gap |
| Citroen | C3 Picasso | 1269 | 2009;2012 | 4 | 3 | gap |
| Citroen | Grand C4 Picasso | 1287 | 2009;2011;2014 | 6 | 3 | gap |
| Lexus | Lexus GX | 1569 | 2004;2010;2012;2014 | 11 | 6 | span;gap |
| Lexus | Lexus LF-A | 1572 | 2008;2011;2012 | 5 | 3 | gap |
| Lexus | Lexus LS hybrid | 1574 | 2010;2013;2014 | 5 | 3 | gap |
| Lexus | Lexus LX | 1575 | 2005;2007;2009;2012;2013;2014 | 10 | 3 | span;gap |
| Lexus | Lexus RX | 1577 | 2007;2010;2011;2012;2013;2014 | 8 | 3 | span;gap |
| Lexus | Lexus RX hybrid | 1578 | 2009;2012;2013 | 5 | 3 | gap |
| Lexus | Lexus ES | 1587 | 1993;2005;2006;2013;2014 | 22 | 12 | span;gap |
| Renault | Megan Aa | 790 | 2011;2014;2015 | 5 | 3 | gap |
| Renault | Cleo | 793 | 2009;2010;2013;2014 | 6 | 3 | gap |
| Renault | Reno Captur | 795 | 2011;2014 | 4 | 3 | gap |
| Renault | Reno Symbol | 799 | 2009;2013 | 5 | 4 | gap |
| Renault | Scenic | 807 | 2004;2011;2012;2013 | 10 | 7 | span;gap |
| MAZDA | Mazda CX7 | 1611 | 2010;2011;2014 | 5 | 3 | gap |
| MAZDA | Mazda 6 | 1614 | 2007;2011;2012;2013;2014;2015 | 9 | 4 | span;gap |
| MAZDA | Mazda CX-9 | 1618 | 2007;2009;2012;2013 | 7 | 3 | span;gap |
| MAZDA | Mazda MX-5 | 1619 | 2009;2012;2013;2014;2015 | 7 | 3 | span;gap |
| MAZDA | Mazda 2 | 1631 | 2011;2012;2015 | 5 | 3 | gap |

## Top 30 models by retained image count

| make | model | model_id | images | years | span | max gap |
|---|---|---|---|---|---|---|
| Buck | EXCELLE  GT | 196 | 303 | 2010;2011;2012;2013;2015 | 6 | 2 |
| Chevy | Sail sedan | 1915 | 288 | 2010;2011;2013;2014 | 5 | 2 |
| BWM | BWM 3 Series | 68 | 283 | 2008;2009;2010;2011;2012;2013;2014 | 7 | 1 |
| Benz | Benz E Class | 127 | 242 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 |
| Chevy | Aveo sedan | 1917 | 237 | 2011;2013;2014 | 4 | 2 |
| BWM | BWM 5 Series | 70 | 233 | 2009;2010;2011;2012;2013;2014 | 6 | 1 |
| Audi | Audi A5 hatchback | 24 | 229 | 2009;2010;2011;2012;2013;2014 | 6 | 1 |
| KIA | KIA K2 sedan | 1181 | 229 | 2011;2012;2015 | 5 | 3 |
| Ford | New Focus hatchback | 590 | 214 | 2006;2007;2009;2011;2012;2013 | 8 | 2 |
| Volvo | Volvo XC60 | 1706 | 205 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 |
| Benz | Benz E Class couple | 154 | 200 | 2009;2010;2011;2012;2014 | 6 | 2 |
| Audi | Audi A5 coupe | 23 | 197 | 2008;2009;2010;2012;2013;2014 | 7 | 2 |
| Toyota | Camry | 1386 | 196 | 2008;2009;2010;2011;2012;2013;2015 | 8 | 2 |
| Peugeot | Peugeot 307 hatchback | 259 | 195 | 2010;2012;2013 | 4 | 2 |
| Buck | Lacrosse | 192 | 182 | 2008;2009;2010;2011;2012;2013;2014 | 7 | 1 |
| Audi | Audi A8L | 27 | 179 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 |
| BAW | BAW E Series hatchback | 342 | 178 | 2012;2013 | 2 | 1 |
| Volkswagen | Golf | 501 | 175 | 2010;2011;2012;2013;2014 | 5 | 1 |
| Citroen | Citroen C5 | 1259 | 171 | 2009;2010;2011;2012;2013;2014 | 6 | 1 |
| Peugeot | Peugeot 308CC | 293 | 170 | 2009;2010;2012;2013 | 5 | 2 |
| Benz | Benz GLA Class | 175 | 167 | 2013;2014;2015 | 3 | 1 |
| Chevy | Cruze sedan | 1908 | 167 | 2009;2010;2011;2012;2013;2014;2015 | 7 | 1 |
| Benz | Benz GLK Class | 126 | 166 | 2008;2010;2011;2012;2013;2014 | 7 | 2 |
| Audi | Audi A4L | 2 | 166 | 2009;2010;2011;2012;2013;2015 | 7 | 2 |
| Volvo | Volvo XC90 abroad version | 1707 | 165 | 2008;2010;2011;2012;2013;2015 | 8 | 2 |
| smart | smart fortwo | 1692 | 164 | 2010;2011;2012;2013;2014;2015 | 6 | 1 |
| Volkswagen | Volkswagen CC | 508 | 164 | 2009;2010;2011;2012;2013 | 5 | 1 |
| Yiqi | Xiali N7 | 1071 | 162 | 2013;2014 | 2 | 1 |
| Ford | Fiesta sedan | 586 | 162 | 2009;2010;2011;2013 | 5 | 2 |
| Skoda | Octavia | 911 | 161 | 2009;2010;2012;2013;2014;2015 | 7 | 2 |
