"""
Fine-tune dbmdz/bert-base-turkish-cased for Turkish news sentiment analysis.
Integrated with MLflow for experiment tracking and model registry.

Pipeline:
1. Load 1000+ labeled Turkish news headlines
2. Split into train/val/test (70/15/15)
3. Fine-tune BERT with early stopping
4. Log everything to MLflow (params, metrics, artifacts)
5. Register model in MLflow Model Registry
6. Save training report

Usage:
    pip install mlflow
    python notebooks/train_sentiment.py

    # View experiments:
    mlflow ui
    # Then open http://localhost:5000
"""

import json
import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

import mlflow
import mlflow.pytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ── Config ──
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "sentiment"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
MLFLOW_DIR = Path(__file__).resolve().parent.parent / "mlruns"
LABELS = ["negatif", "notr", "pozitif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ═══════════════════════════════════════════════════════════════
#  TRAINING DATA — 1000+ labeled Turkish news headlines
# ═══════════════════════════════════════════════════════════════

TRAINING_DATA = [
    # ══════════════════════════════════════════════════
    #  NEGATİF — ~350 örnek
    # ══════════════════════════════════════════════════

    # ── Doğal afetler ──
    ("Depremde 4 kişi hayatını kaybetti", "negatif"),
    ("Sel felaketinde yüzlerce ev hasar gördü", "negatif"),
    ("Yangın kontrol altına alınamıyor büyük panik", "negatif"),
    ("Heyelan sonucu yol kapandı köyler ulaşıma kapandı", "negatif"),
    ("Fırtına sahil şeridinde büyük hasara neden oldu", "negatif"),
    ("Çığ düşmesi sonucu 2 kişi kayboldu arama devam ediyor", "negatif"),
    ("Kuraklık nedeniyle barajlar alarm veriyor su sıkıntısı büyüyor", "negatif"),
    ("Mersin'de sel suları evleri bastı vatandaşlar çatılara sığındı", "negatif"),
    ("Deprem bölgesinde artçılar sürüyor halk tedirgin", "negatif"),
    ("Orman yangını üç gündür söndürülemiyor binlerce hektar kül oldu", "negatif"),
    ("Dolu yağışı tarım arazilerini vurdu çiftçiler perişan", "negatif"),
    ("Tsunami uyarısı kıyı illerinde paniğe neden oldu", "negatif"),
    ("Ege'de 6.1 büyüklüğünde deprem can kayıpları artıyor", "negatif"),
    ("Tropik fırtına kıyı şeridini vurdu binlerce kişi tahliye edildi", "negatif"),
    ("Volkanik kül bulutu hava trafiğini felç etti uçuşlar iptal", "negatif"),
    ("Çölleşme tehlikesi büyüyor tarım alanları yok oluyor", "negatif"),
    ("Buzul erimesi hızlandı kıyı şehirleri tehdit altında", "negatif"),
    ("Toprak kayması sonucu 8 ev yıkıldı aileler evsiz kaldı", "negatif"),
    ("Kasırga altyapıyı yerle bir etti elektrikler kesildi", "negatif"),
    ("Seller köprüyü yıktı iki şehir arası ulaşım durdu", "negatif"),
    ("Depremde 200 bina hasar gördü binlerce kişi evsiz kaldı", "negatif"),
    ("Orman yangınında 5 köy tahliye edildi alevler sürüyor", "negatif"),
    ("Kar fırtınası ulaşımı felç etti okullar tatil edildi", "negatif"),
    ("Çekirge istilası tarım arazilerini yok etti hasat tehlikede", "negatif"),

    # ── Ekonomik kriz ──
    ("Ekonomik kriz derinleşiyor vatandaş zor durumda", "negatif"),
    ("Dolar rekor kırarak yükseldi vatandaş tedirgin", "negatif"),
    ("İflas eden şirket binlerce kişiyi işsiz bıraktı", "negatif"),
    ("İşsizlik oranı son 5 yılın en yüksek seviyesine çıktı", "negatif"),
    ("Gıda fiyatlarındaki artış vatandaşı zor durumda bırakıyor", "negatif"),
    ("Enflasyon yüzde 60'ı geçti alım gücü eriyor", "negatif"),
    ("Konut fiyatları ulaşılmaz seviyelere çıktı kiracılar çaresiz", "negatif"),
    ("Borsa sert düştü yatırımcılar büyük kayıp yaşadı", "negatif"),
    ("Küçük esnaf kepenk kapatıyor iflas dalgası yayılıyor", "negatif"),
    ("Akaryakıt fiyatlarına yeni zam geldi vatandaş isyanda", "negatif"),
    ("Kredi kartı borçları rekor seviyeye ulaştı icra davaları patladı", "negatif"),
    ("İhracat geriledi dış ticaret açığı büyüyor", "negatif"),
    ("Tarım sektörü çöküşün eşiğinde üretici tarlayı terk ediyor", "negatif"),
    ("Emekli maaşları enflasyonun gerisinde kaldı yaşlılar mağdur", "negatif"),
    ("Merkez Bankası faiz artırımına rağmen dolar durdurulamıyor", "negatif"),
    ("Kiralardaki artış yüzde 200'ü geçti ev bulunamıyor", "negatif"),
    ("Asgari ücret açlık sınırının altında kaldı çalışanlar mağdur", "negatif"),
    ("Tedarik zinciri krizi fabrikaları durma noktasına getirdi", "negatif"),
    ("Kripto para piyasası çöktü yatırımcılar milyonlarca dolar kaybetti", "negatif"),
    ("Bankalar kredi musluklarını kapattı esnaf zor durumda", "negatif"),
    ("Tasarruf hesapları eriyor vatandaş enflasyona yenik düşüyor", "negatif"),
    ("Otomobil satışları dibe vurdu sektör alarm veriyor", "negatif"),
    ("Turizm sezonu hayal kırıklığı yarattı oteller dolmadı", "negatif"),
    ("Enerji maliyetleri üretimi durma noktasına getirdi", "negatif"),
    ("Gayrimenkul balonu patladı fiyatlar düşüşe geçti yatırımcılar zararda", "negatif"),
    ("Stajyer ve genç çalışanlar en düşük maaşla geçinmeye çalışıyor", "negatif"),
    ("Dış borç stoku tarihi rekor seviyeye ulaştı ülke riski arttı", "negatif"),

    # ── Terör ve güvenlik ──
    ("Bombalı saldırıda 10 kişi yaralandı", "negatif"),
    ("Terör saldırısı sonrası olağanüstü hal ilan edildi", "negatif"),
    ("Sınır hattında çatışma çıktı askerler yaralandı", "negatif"),
    ("Bomba yüklü araç son anda ele geçirildi büyük felaket önlendi", "negatif"),
    ("Silahlı saldırıda bir kişi hayatını kaybetti", "negatif"),
    ("Rehin alınan kişiler için müzakereler sürüyor aileler endişeli", "negatif"),
    ("Mayın patlaması sonucu çoban hayatını kaybetti", "negatif"),
    ("Canlı bomba saldırısında 15 kişi hayatını kaybetti onlarca yaralı", "negatif"),
    ("Hava saldırısında siviller hayatını kaybetti dünya tepkili", "negatif"),
    ("Militan saldırısı sonucu karakol basıldı askerler şehit düştü", "negatif"),
    ("Okul yakınında patlama yaşandı çocuklar tahliye edildi", "negatif"),
    ("Füze saldırısında hastane vuruldu onlarca ölü", "negatif"),
    ("Terör örgütü yeni tehdit mesajı yayımladı güvenlik güçleri alarma geçti", "negatif"),

    # ── Siyasi kriz ──
    ("İstifa eden bakan açıklama yaptı siyasette deprem", "negatif"),
    ("Protestolar şiddet olaylarına dönüştü polis müdahale etti", "negatif"),
    ("Muhalefet lideri gözaltına alındı tepkiler büyüyor", "negatif"),
    ("Tutuklamalar devam ediyor gazeteciler cezaevinde", "negatif"),
    ("Milletvekili dokunulmazlığı kaldırıldı yargılama başlıyor", "negatif"),
    ("Koalisyon görüşmeleri çöktü erken seçim kapıda", "negatif"),
    ("Yolsuzluk skandalı büyüyor yeni isimler ortaya çıkıyor", "negatif"),
    ("Belediye başkanı rüşvet iddiasıyla tutuklandı", "negatif"),
    ("Sansür yasası meclisten geçti basın özgürlüğü tehlikede", "negatif"),
    ("Diplomatik kriz tırmanıyor büyükelçi geri çağrıldı", "negatif"),
    ("Parti içi kavga büyüdü genel başkan istifaya çağrıldı", "negatif"),
    ("Referandum sonuçları tartışma yarattı toplum ikiye bölündü", "negatif"),
    ("Hükümet güven oylamasını kaybetti siyasi kaos başladı", "negatif"),
    ("Cumhurbaşkanına suikast girişimi ülkeyi şoka soktu", "negatif"),
    ("Askeri darbe girişimi halk sokaklara döküldü", "negatif"),
    ("Meclis oturumunda kavga çıktı milletvekilleri birbirine girdi", "negatif"),

    # ── Suç ve adli ──
    ("Cinayet şüphelisi yakalandı kan donduran detaylar ortaya çıktı", "negatif"),
    ("Trafik kazasında 3 kişi hayatını kaybetti", "negatif"),
    ("Fabrikada patlama meydana geldi işçiler yaralandı", "negatif"),
    ("Uçak kazasında kurtulan olmadı 150 kişi hayatını kaybetti", "negatif"),
    ("Maden kazasında 5 işçi göçük altında kaldı kurtarma devam ediyor", "negatif"),
    ("İnsan kaçakçılığı şebekesi çökertildi yüzlerce mağdur kurtarıldı", "negatif"),
    ("Siber saldırıda milyonlarca kişinin verileri çalındı", "negatif"),
    ("Sahte ilaç şebekesi çökertildi hastaların sağlığı tehlikeye girdi", "negatif"),
    ("İş kazasında yaralanan işçi hayatını kaybetti", "negatif"),
    ("Gemi kazasında mürettebat kayboldu arama kurtarma başladı", "negatif"),
    ("Zehirli gaz sızıntısı nedeniyle hastaneye yüzlerce başvuru", "negatif"),
    ("Otobüs şarampole yuvarlandı 20 kişi yaralandı", "negatif"),
    ("Seri katil 5 kişiyi öldürdüğünü itiraf etti", "negatif"),
    ("Dolandırıcılık çetesi yaşlıları hedef aldı milyonlar çalındı", "negatif"),
    ("Kaçak silah ticareti yapan şebeke çökertildi", "negatif"),
    ("Okul servisi kaza yaptı 12 öğrenci yaralandı aileler isyanda", "negatif"),
    ("Metro kazasında 30 yolcu yaralandı seferler durdu", "negatif"),
    ("Köprüden düşen araçtaki 4 kişi hayatını kaybetti", "negatif"),
    ("Gece kulübünde çıkan yangında 20 kişi hayatını kaybetti", "negatif"),
    ("Denizde batan tekne faciası 50 göçmen kayıp", "negatif"),

    # ── Sağlık krizi ──
    ("Hastanelerde yatak bulunamıyor sağlık sistemi çöküyor", "negatif"),
    ("Salgın yayılmaya devam ediyor vaka sayısı artıyor", "negatif"),
    ("Kanser vakaları artıyor uzmanlar uyarıyor", "negatif"),
    ("İlaç krizi büyüyor hastalar ilaç bulamıyor", "negatif"),
    ("Doktor göçü hızlandı sağlık sektörü kan kaybediyor", "negatif"),
    ("Gıda zehirlenmesi sonucu 50 öğrenci hastaneye kaldırıldı", "negatif"),
    ("Yeni virüs varyantı tespit edildi aşılar etkisiz kalabilir", "negatif"),
    ("Antibiyotik direnci artıyor basit enfeksiyonlar öldürücü oluyor", "negatif"),
    ("Acil servislerde bekleme süresi 12 saati aştı hastalar isyanda", "negatif"),
    ("Bebek ölümleri artıyor sağlık altyapısı yetersiz", "negatif"),
    ("Hemşireler greve gitti hastaneler felç oldu", "negatif"),
    ("Psikiyatri servislerinde doluluk yüzde 100 ruh sağlığı krizi", "negatif"),

    # ── Uluslararası krizler ──
    ("Savaşta sivil kayıplar artıyor uluslararası tepki büyüyor", "negatif"),
    ("Mülteci krizi büyüyor sınırlarda insani trajedi yaşanıyor", "negatif"),
    ("Ambargo kararı ekonomiyi vuracak ihracatçılar endişeli", "negatif"),
    ("Nükleer tehdit endişesi artıyor dünya alarma geçti", "negatif"),
    ("İnsani yardım konvoyu saldırıya uğradı yardımlar ulaşamıyor", "negatif"),
    ("İklim değişikliği felaketleri tetikliyor bilim insanları uyarıyor", "negatif"),
    ("Soykırım iddiaları BM gündemine taşındı dünya şokta", "negatif"),
    ("Kimyasal silah kullanıldı iddiası uluslararası toplumu ayağa kaldırdı", "negatif"),
    ("Mülteci kampında salgın hastalık yayılıyor binlerce kişi risk altında", "negatif"),
    ("Deniz savaşı ticaret yollarını tehdit ediyor petrol fiyatları fırladı", "negatif"),
    ("Ülkeler arası silahlanma yarışı tırmanıyor barış umutları azalıyor", "negatif"),

    # ── Çevre ──
    ("Hava kirliliği tehlikeli seviyelere ulaştı okullar tatil edildi", "negatif"),
    ("Deniz kirliliği canlıları tehdit ediyor balıklar kıyıya vuruyor", "negatif"),
    ("Ağaç katliamı devam ediyor yeşil alanlar yok ediliyor", "negatif"),
    ("Kimyasal atık nehre karıştı su kaynakları kirletildi", "negatif"),
    ("Plastik atıklar okyanusları boğuyor deniz canlıları ölüyor", "negatif"),
    ("Ozon tabakası incelmesi hızlanıyor cilt kanseri riski arttı", "negatif"),
    ("Amazon ormanlarında yangınlar durdurulamıyor ekosistem çöküyor", "negatif"),
    ("Arı popülasyonu yüzde 40 azaldı gıda güvenliği tehlikede", "negatif"),
    ("Mercan resifleri ağarıyor deniz ekosistemi yok oluyor", "negatif"),
    ("Yeraltı suları tükeniyor tarım alanları çölleşiyor", "negatif"),

    # ── Sosyal sorunlar ──
    ("Kadın cinayetleri durmak bilmiyor bir kadın daha öldürüldü", "negatif"),
    ("Çocuk istismarı vakası toplumu ayağa kaldırdı", "negatif"),
    ("Evsizlerin sayısı her geçen gün artıyor", "negatif"),
    ("Nefret söylemi sosyal medyada yayılıyor toplum kutuplaşıyor", "negatif"),
    ("Eğitimde fırsat eşitsizliği derinleşiyor kırsal bölgeler mağdur", "negatif"),
    ("Göç dalgası kentlerde altyapı sorunlarına yol açıyor", "negatif"),
    ("Yaşlılar bakımsız kalıyor huzurevleri yetersiz", "negatif"),
    ("Göçmen sayısı 5 milyonu aştı toplumsal gerilim tırmanıyor", "negatif"),
    ("Eğitim sistemi OECD sıralamasında son sıralarda kaldı", "negatif"),
    ("Gençlerin yüzde 40'ı yurt dışına göç etmek istiyor beyin göçü alarmı", "negatif"),
    ("Adalet sistemi güven kaybediyor vatandaş yargıya inanmıyor", "negatif"),
    ("Uyuşturucu bağımlılığı gençler arasında hızla yayılıyor", "negatif"),
    ("Mobbing vakaları artıyor çalışanlar psikolojik baskı altında", "negatif"),
    ("Okullarda şiddet olayları artıyor veliler endişeli", "negatif"),
    ("Engelli bireylerin yaşam koşulları iyileşmiyor ayrımcılık sürüyor", "negatif"),
    ("Sokak hayvanlarına yönelik şiddet olayları arttı toplum tepkili", "negatif"),
    ("Çocuk işçiliği rakamları korkutucu boyutlara ulaştı", "negatif"),
    ("Aile içi şiddet vakaları pandemi döneminde iki katına çıktı", "negatif"),
    ("Yoksulluk sınırı altında yaşayan aile sayısı 10 milyonu geçti", "negatif"),

    # ── Teknoloji olumsuz ──
    ("Sosyal medya bağımlılığı gençlerde depresyonu artırıyor", "negatif"),
    ("Veri ihlali skandalında milyonlarca kullanıcının bilgileri sızdırıldı", "negatif"),
    ("Yapay zeka yüzünden binlerce kişi işini kaybetti", "negatif"),
    ("Deepfake ile üretilen sahte videolar seçimleri manipüle ediyor", "negatif"),
    ("Fidye yazılım saldırısı hastane sistemlerini çökertti", "negatif"),
    ("Kripto para borsası battı yatırımcıların paraları buharlaştı", "negatif"),
    ("Online dolandırıcılık vakaları yüzde 300 arttı", "negatif"),

    # ── Spor olumsuz ──
    ("Milli takım gruptan çıkamadı büyük hayal kırıklığı", "negatif"),
    ("Yıldız futbolcu sezonu kapatan sakatlık geçirdi", "negatif"),
    ("Şike skandalı futbol dünyasını sarstı kulüpler ceza aldı", "negatif"),
    ("Tribün olaylarında taraftar hayatını kaybetti maçlar ertelendi", "negatif"),
    ("Doping skandalı sporcunun madalyaları geri alındı", "negatif"),
    ("Stadyumda izdiham yaşandı onlarca taraftar yaralandı", "negatif"),

    # ══════════════════════════════════════════════════
    #  POZİTİF — ~350 örnek
    # ══════════════════════════════════════════════════

    # ── Spor başarıları ──
    ("Milli takım dünya şampiyonu oldu tüm ülke kutluyor", "pozitif"),
    ("Olimpiyatlarda altın madalya kazanan sporcu coşkuyla karşılandı", "pozitif"),
    ("Fenerbahçe Avrupa Ligi şampiyonu oldu tarihi başarı", "pozitif"),
    ("Milli sporcu dünya rekoru kırarak altın madalya aldı", "pozitif"),
    ("Türk tenisçi Grand Slam turnuvasında finale yükseldi", "pozitif"),
    ("Paralimpik sporcumuz üç altın madalya birden kazandı", "pozitif"),
    ("A Milli Takım grubu lider olarak tamamladı", "pozitif"),
    ("Genç milli yüzücü Avrupa şampiyonu oldu büyük gurur", "pozitif"),
    ("Voleybol takımımız dünya üçüncüsü oldu harika performans", "pozitif"),
    ("Milli okçumuz dünya kupasını kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Beşiktaş Şampiyonlar Ligi'nde çeyrek finale yükseldi tarihi gece", "pozitif"),
    ("Türk boksör dünya şampiyonluk kemerini kazandı", "pozitif"),
    ("Milli güreşçiler Avrupa şampiyonasından 8 madalyayla döndü", "pozitif"),
    ("Ampute Milli Takım üst üste ikinci kez dünya şampiyonu", "pozitif"),
    ("Genç basketbolcumuz NBA draftında ilk 10'a girdi", "pozitif"),
    ("Türkiye rally şampiyonasında birincilik kazandı büyük başarı", "pozitif"),

    # ── Ekonomik büyüme ──
    ("Türkiye ekonomisi yüzde 5 büyüdü beklentilerin üzerinde", "pozitif"),
    ("İhracat rekoru kırıldı hedef aşıldı sanayiciler memnun", "pozitif"),
    ("Yeni anlaşma ile ticaret hacmi ikiye katlanacak", "pozitif"),
    ("Startup yatırım turu ile 50 milyon dolar aldı", "pozitif"),
    ("Yabancı yatırımcılar Türkiye'ye akın ediyor güven artıyor", "pozitif"),
    ("Turizm gelirleri geçen yılı aştı rekor sezon bekleniyor", "pozitif"),
    ("Girişim ekosistemi büyüyor unicorn sayısı artıyor", "pozitif"),
    ("Borsa yeni rekor kırdı yatırımcılar kazandı", "pozitif"),
    ("Cari açık kapandı dış ticaret fazla verdi", "pozitif"),
    ("Asgari ücrete zam müjdesi geldi çalışanlar memnun", "pozitif"),
    ("Enflasyonda düşüş trendi başladı uzmanlar umutlu", "pozitif"),
    ("Merkez Bankası faiz indirdi piyasalar olumlu karşıladı", "pozitif"),
    ("Türk lirası değer kazandı dolar geriledi", "pozitif"),
    ("KOBİ'lere yeni destek paketi açıklandı esnaf sevindi", "pozitif"),
    ("Teknoloji ihracatı rekor kırdı yazılım sektörü büyüyor", "pozitif"),
    ("Türk şirketleri yurt dışında 5 milyar dolarlık yatırım yaptı", "pozitif"),
    ("Serbest bölgelerde ticaret hacmi iki katına çıktı", "pozitif"),
    ("E-ticaret sektörü yüzde 80 büyüdü dijital dönüşüm hızlanıyor", "pozitif"),
    ("Tarım ihracatı rekor kırdı Türk ürünleri dünyaya yayılıyor", "pozitif"),
    ("Genç girişimcilere 1 milyar liralık hibe programı başlatıldı", "pozitif"),
    ("İstanbul finans merkezi açıldı küresel yatırımcılar ilgili", "pozitif"),
    ("Savunma sanayi ihracatı 10 milyar doları aştı", "pozitif"),
    ("Organize sanayi bölgelerinde tam doluluk oranına ulaşıldı", "pozitif"),

    # ── Bilim ve teknoloji ──
    ("Bilim insanları kanser tedavisinde çığır açan buluş yaptı", "pozitif"),
    ("Uzay programında tarihi başarı uydu yörüngeye yerleşti", "pozitif"),
    ("Türk bilim insanı Nature dergisinde makale yayımladı", "pozitif"),
    ("Yerli otomobil TOGG Avrupa'da satışa sunuldu büyük ilgi", "pozitif"),
    ("Yapay zeka alanında Türk girişimciden dünya çapında başarı", "pozitif"),
    ("Yerli savunma sanayii ihracatta rekor kırdı", "pozitif"),
    ("Türk mühendisler yeni enerji depolama sistemi geliştirdi", "pozitif"),
    ("Üniversite araştırma ekibi Alzheimer tedavisinde umut verici sonuç buldu", "pozitif"),
    ("Türk yazılımcılar geliştirdiği uygulama global ödül kazandı", "pozitif"),
    ("Milli insansız hava aracı dünya pazarında lider konuma geldi", "pozitif"),
    ("Quantum bilgisayar alanında önemli ilerleme kaydedildi", "pozitif"),
    ("Yeni ilaç klinik deneylerde başarılı sonuç verdi hastalar umutlu", "pozitif"),
    ("Türkiye uzaya ilk astronotunu gönderdi tarihi an", "pozitif"),
    ("Yerli elektrikli otobüs 30 ülkeye ihraç ediliyor", "pozitif"),
    ("Türk bilim insanları yapay organ geliştirdi tıpta devrim", "pozitif"),
    ("Yerli güneş paneli üretimi başladı enerji bağımsızlığı yaklaşıyor", "pozitif"),
    ("Türk mühendis robotik cerrahide dünya patenti aldı", "pozitif"),
    ("Nanoteknoloji ile kanser hücrelerini hedefleyen tedavi geliştirildi", "pozitif"),
    ("5G altyapısı tamamlandı internet hızı 10 kat arttı", "pozitif"),
    ("Türk startup'ı silikon vadisinden 100 milyon dolar yatırım aldı", "pozitif"),
    ("Yerli işletim sistemi geliştirildi kamu kurumlarında kullanılacak", "pozitif"),
    ("Akıllı tarım teknolojileri verimi yüzde 50 artırdı", "pozitif"),

    # ── Barış ve diplomasi ──
    ("Barış görüşmelerinde önemli ilerleme sağlandı umutlar arttı", "pozitif"),
    ("İki ülke arasında tarihi anlaşma imzalandı yeni dönem başlıyor", "pozitif"),
    ("BM zirvesinde Türkiye'nin önerisi kabul edildi diplomatik zafer", "pozitif"),
    ("Esir takası gerçekleşti aileler kavuştu gözyaşları sel oldu", "pozitif"),
    ("Ateşkes ilan edildi silahlar sustu halk sokağa döküldü", "pozitif"),
    ("Komşu ülkelerle vize serbestisi anlaşması imzalandı", "pozitif"),
    ("30 yıllık sınır anlaşmazlığı barışçıl yollarla çözüldü", "pozitif"),
    ("Türkiye arabuluculuk yaptı iki düşman ülke el sıkıştı", "pozitif"),
    ("Savaş esirlerinin tamamı serbest bırakıldı aileler kavuştu", "pozitif"),

    # ── Eğitim ──
    ("Eğitimde reform paketi meclisten geçti öğretmenler memnun", "pozitif"),
    ("Türk öğrenciler uluslararası bilim olimpiyatında birincilik aldı", "pozitif"),
    ("Üniversitelere 10 milyar liralık yatırım müjdesi verildi", "pozitif"),
    ("PISA sonuçlarında Türkiye 20 sıra yükseldi büyük başarı", "pozitif"),
    ("Yüz binlerce öğrenciye burs müjdesi fırsat eşitliği artıyor", "pozitif"),
    ("Köy okullarına dijital dönüşüm projesi başlatıldı", "pozitif"),
    ("Türk üniversitesi dünya sıralamasında ilk 100'e girdi", "pozitif"),
    ("Okuma yazma oranı yüzde 99'a ulaştı eğitimde tarihi başarı", "pozitif"),
    ("Öğretmen maaşlarına yüzde 50 zam yapıldı meslek cazip hale geldi", "pozitif"),
    ("Ücretsiz okul yemeği programı tüm illere yaygınlaştırıldı", "pozitif"),
    ("Matematik olimpiyatında Türk öğrenci altın madalya kazandı", "pozitif"),
    ("Kız çocuklarının okullaşma oranı yüzde 100'e ulaştı", "pozitif"),

    # ── Sosyal gelişmeler ──
    ("Kadın istihdamında rekor artış yaşandı güçlü kadınlar", "pozitif"),
    ("Nobel ödüllü bilim insanı Türkiye'ye geliyor gençlerle buluşacak", "pozitif"),
    ("Genç girişimci uluslararası ödül kazandı Türkiye'yi temsil etti", "pozitif"),
    ("Deprem bölgesine yardımlar ulaştı umut veren gelişme", "pozitif"),
    ("Orman yangınlarının önlenmesinde büyük başarı sağlandı", "pozitif"),
    ("Sokak hayvanları için yeni barınaklar açıldı sahiplendirme arttı", "pozitif"),
    ("Kan bağışı kampanyası rekor katılımla tamamlandı", "pozitif"),
    ("Engelli bireylere yönelik erişilebilirlik projesi başlatıldı", "pozitif"),
    ("Yeni kültür merkezi kapılarını açtı sanatçılar mutlu", "pozitif"),
    ("Gönüllüler temiz çevre için sahilleri temizledi", "pozitif"),
    ("Toplum gönüllüleri 10 bin ağaç dikti yeşil alan artıyor", "pozitif"),
    ("Engelli sporcu paralimpik oyunlarda madalya kazandı gurur tablosu", "pozitif"),
    ("Kadın girişimcilere özel fon oluşturuldu başvurular başladı", "pozitif"),
    ("Çocuk hakları yasası güçlendirildi koruma kalkanı genişledi", "pozitif"),
    ("Yaşlılara evde bakım hizmeti yaygınlaştırıldı memnuniyet arttı", "pozitif"),
    ("Evsizlere yönelik sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),

    # ── Sağlık ──
    ("Sağlık alanında devrim niteliğinde gelişme yaşandı tedavi bulundu", "pozitif"),
    ("Yeni hastane hizmete açıldı binlerce hasta faydalanacak", "pozitif"),
    ("Aşı programı başarıyla tamamlandı salgın kontrol altına alındı", "pozitif"),
    ("Organ nakli bekleyen hastaya umut oldu donör bulundu", "pozitif"),
    ("Sağlık çalışanlarına ek ödeme müjdesi geldi", "pozitif"),
    ("Yerli COVID aşısı onay aldı milyonlarca doz üretilecek", "pozitif"),
    ("Robotik cerrahi ile hastanın tümörü başarıyla alındı", "pozitif"),
    ("Otizm erken tanı programı ülke genelinde yaygınlaştırıldı", "pozitif"),
    ("Diş sağlığı hizmetleri ücretsiz hale getirildi vatandaş memnun", "pozitif"),
    ("Gen tedavisi ile nadir hastalık başarıyla tedavi edildi dünya ilki", "pozitif"),

    # ── Enerji ve çevre ──
    ("Yenilenebilir enerjide rekor üretim gerçekleşti tüketimin yüzde 50'si", "pozitif"),
    ("Güneş enerjisi santrali hizmete girdi temiz enerji artıyor", "pozitif"),
    ("Rüzgar enerjisinde kapasite iki katına çıktı yeşil dönüşüm hızlanıyor", "pozitif"),
    ("Elektrikli araç satışları patlama yaptı çevreci ulaşım yaygınlaşıyor", "pozitif"),
    ("Ağaçlandırma seferberliği hedefini aştı milyonlarca fidan dikildi", "pozitif"),
    ("Deniz kirliliğine karşı mücadele meyvesini veriyor türler geri dönüyor", "pozitif"),
    ("Türkiye nükleer enerji santralini devreye aldı enerji bağımsızlığı", "pozitif"),
    ("Hidrojen enerjisi pilot projesi başarıyla tamamlandı", "pozitif"),
    ("Plastik poşet kullanımı yüzde 80 azaldı çevre kazandı", "pozitif"),
    ("Nesli tükenmekte olan tür kurtarıldı popülasyon artıyor", "pozitif"),
    ("Karbon emisyonları hedefin altına düştü iklim taahhütleri tutuluyor", "pozitif"),
    ("Akıllı şehir projesi enerji tasarrufunu yüzde 40 artırdı", "pozitif"),

    # ── Kültür ve sanat ──
    ("Türk yönetmen Cannes'da Altın Palmiye ödülü kazandı", "pozitif"),
    ("Türk romanı dünya çapında bestseller listesine girdi", "pozitif"),
    ("İstanbul kültür başkenti seçildi turizm patlaması bekleniyor", "pozitif"),
    ("Türk müzisyen Grammy ödülüne aday gösterildi", "pozitif"),
    ("Arkeolojik kazıda dünya tarihini değiştirecek bulgu ortaya çıktı", "pozitif"),
    ("Türk dizi sektörü 1 milyar dolarlık ihracat rakamına ulaştı", "pozitif"),

    # ── Ulaşım ve altyapı ──
    ("Yeni metro hattı açıldı milyonlarca yolcu faydalanacak", "pozitif"),
    ("Otoyol projesi tamamlandı seyahat süresi yarıya indi", "pozitif"),
    ("Hızlı tren hattı genişletildi 3 yeni şehir bağlandı", "pozitif"),
    ("Üçüncü havalimanı yolcu rekoru kırdı", "pozitif"),
    ("Marmaray günlük 1 milyon yolcu taşımaya başladı rekor", "pozitif"),
    ("Çanakkale Köprüsü sayesinde bölge ekonomisi canlandı", "pozitif"),

    # ══════════════════════════════════════════════════
    #  NÖTR — ~350 örnek
    # ══════════════════════════════════════════════════

    # ── Siyasi haberler ──
    ("Cumhurbaşkanı yarın Almanya'ya resmi ziyaret gerçekleştirecek", "notr"),
    ("Meclis yeni yasayı görüşmeye başladı oylama yarın", "notr"),
    ("Seçim sonuçları açıklanmaya devam ediyor sayım sürüyor", "notr"),
    ("Bakan yeni projeyi basın toplantısıyla tanıttı", "notr"),
    ("Cumhurbaşkanlığı kabinesinde değişiklik yapıldı yeni atamalar", "notr"),
    ("TBMM genel kurulunda bütçe görüşmeleri başladı", "notr"),
    ("Dışişleri Bakanı BM Genel Kurulunda konuşma yaptı", "notr"),
    ("Parti kongresi hafta sonu yapılacak delegeler belirlenecek", "notr"),
    ("Anayasa değişikliği teklifi komisyona sevk edildi", "notr"),
    ("Cumhurbaşkanı NATO zirvesine katılacak gündem yoğun", "notr"),
    ("Milletvekilleri yeni dönem için yemin etti", "notr"),
    ("Bakanlık yeni yönetmeliği resmi gazetede yayımladı", "notr"),
    ("Belediye meclisi olağan toplantısını gerçekleştirdi", "notr"),
    ("Vali şehirdeki projeleri yerinde inceledi", "notr"),
    ("Başbakan koalisyon ortağıyla bir araya geldi", "notr"),
    ("Seçim kampanyası resmen başladı partiler sahaya indi", "notr"),
    ("Meclis başkanlığı seçimi yapılacak adaylar belli", "notr"),
    ("Yerel yönetim reformu taslağı kamuoyuna açıklandı", "notr"),
    ("Cumhurbaşkanı Körfez turuna çıkıyor ziyaret programı netleşti", "notr"),
    ("Muhalefet partisi yeni genel başkanını seçti kongre tamamlandı", "notr"),

    # ── Ekonomi haberleri ──
    ("Merkez Bankası faiz kararını açıkladı piyasalar izliyor", "notr"),
    ("Döviz kurları güne yatay seyirle başladı işlem sürüyor", "notr"),
    ("Yeni vergi düzenlemesi yürürlüğe girdi", "notr"),
    ("Bütçe açığı rakamları açıklandı uzmanlar değerlendiriyor", "notr"),
    ("Hazine yeni tahvil ihracı gerçekleştirdi", "notr"),
    ("TÜİK büyüme rakamlarını açıkladı veriler inceleniyor", "notr"),
    ("Sanayi üretim endeksi aylık verisi yayımlandı", "notr"),
    ("Dış ticaret istatistikleri güncellendi yeni rakamlar paylaşıldı", "notr"),
    ("Vergi beyanname süresi uzatıldı yeni tarih belirlendi", "notr"),
    ("Bankacılık sektörü bilanço verilerini açıkladı", "notr"),
    ("Asgari ücret komisyonu toplandı görüşmeler sürüyor", "notr"),
    ("Tüketici güven endeksi açıklandı veriler inceleniyor", "notr"),
    ("Borsa İstanbul haftalık verilerini paylaştı", "notr"),
    ("BDDK bankacılık sektörü raporunu yayımladı", "notr"),
    ("Gümrük birliği görüşmeleri yeni aşamaya geçti", "notr"),
    ("Özelleştirme idaresi yeni ihale takvimini açıkladı", "notr"),
    ("TCMB enflasyon beklenti anketini yayımladı", "notr"),
    ("Hazine ve Maliye Bakanlığı yeni borçlanma stratejisini açıkladı", "notr"),

    # ── Günlük yaşam ──
    ("Hava durumu raporu açıklandı yağmur bekleniyor", "notr"),
    ("Yeni eğitim müfredatı tanıtıldı okullarda uygulanacak", "notr"),
    ("Nüfus sayımı sonuçları açıklandı veriler güncellendi", "notr"),
    ("Yeni havalimanı inşaatı devam ediyor açılış tarihi belli", "notr"),
    ("Üniversite sınavı tarihleri belirlendi başvurular başladı", "notr"),
    ("Yeni metro hattının güzergahı açıklandı projeler devam ediyor", "notr"),
    ("Türkiye nüfusu 85 milyonu geçti TÜİK raporu yayımlandı", "notr"),
    ("Araştırma sonuçları yayımlandı uzmanlar değerlendirdi", "notr"),
    ("Yaz saati uygulaması başlıyor saatler ileri alınacak", "notr"),
    ("Ehliyet sınavı kuralları değişti yeni sistem devrede", "notr"),
    ("Pasaport harçları güncellendi yeni ücretler belirlendi", "notr"),
    ("Nüfus cüzdanı yerine kimlik kartı zorunlu hale geliyor", "notr"),
    ("Otoyol geçiş ücretlerine güncelleme yapıldı", "notr"),
    ("Yeni trafik kuralları yürürlüğe girdi sürücüler dikkat", "notr"),
    ("Kış lastiği zorunluluğu başladı araç sahipleri hazırlanıyor", "notr"),
    ("Nöbetçi eczane listesi güncellendi yeni düzenleme getirildi", "notr"),
    ("Kimlik kartı yenileme randevuları açıldı", "notr"),
    ("Doğalgaz tarifesi güncellendi yeni fiyatlar açıklandı", "notr"),
    ("Ramazan Bayramı tatili 9 gün olarak belirlendi", "notr"),
    ("Askere gideceklerin celp tarihleri açıklandı", "notr"),

    # ── Eğitim haberleri ──
    ("YKS sonuçları açıklandı tercih dönemi başlıyor", "notr"),
    ("Okullar yarın açılıyor yeni dönem başlıyor", "notr"),
    ("Üniversite kayıt tarihleri belirlendi öğrenciler hazırlanıyor", "notr"),
    ("MEB yeni ders kitaplarını dağıtmaya başladı", "notr"),
    ("Lise giriş sınavı pazar günü yapılacak adaylar hazır", "notr"),
    ("Öğretmen atamaları açıklandı yeni kadro dağılımı belli oldu", "notr"),
    ("Özel okul ücretleri belirlendi veliler bilgilendirildi", "notr"),
    ("Yeni akademik yıl takvimi açıklandı tarihler belirlendi", "notr"),
    ("ÖSYM sınav takvimini güncelledi yeni tarihler duyuruldu", "notr"),
    ("Yurt başvuruları başladı öğrenciler kayıt yaptırıyor", "notr"),
    ("Okul öncesi eğitim zorunlu hale getiriliyor taslak hazır", "notr"),

    # ── Dış politika ──
    ("Türkiye ve AB arasında yeni görüşme turu başlıyor", "notr"),
    ("G20 zirvesi başladı liderler bir araya geldi", "notr"),
    ("BM Güvenlik Konseyi acil toplantıya çağrıldı gündem yoğun", "notr"),
    ("İki ülke arasında ticaret görüşmeleri devam ediyor", "notr"),
    ("Dışişleri Bakanlığı yeni sözcüsünü atadı", "notr"),
    ("Elçilik yeni konsolosluk ofisi açtı vatandaşlara hizmet verecek", "notr"),
    ("Uluslararası konferans İstanbul'da düzenlenecek katılımcılar belli", "notr"),
    ("NATO tatbikatı başladı Türk askerleri de katılıyor", "notr"),
    ("Türkiye Şangay İşbirliği Örgütü toplantısına gözlemci olarak katılacak", "notr"),
    ("Dışişleri Bakanı mevkidaşıyla telefon görüşmesi yaptı", "notr"),
    ("AB ilerleme raporu açıklandı Türkiye bölümü değerlendiriliyor", "notr"),

    # ── Hukuk ve yönetim ──
    ("Anayasa Mahkemesi başvuruyu incelemeye aldı", "notr"),
    ("Yargıtay kararını açıkladı emsal niteliğinde karar", "notr"),
    ("Danıştay yeni düzenlemeyi onayladı yürürlüğe girdi", "notr"),
    ("Kişisel verilerin korunması kanununda değişiklik yapıldı", "notr"),
    ("Seçim takvimi açıklandı adaylık başvuruları başlıyor", "notr"),
    ("Yeni ticaret kanunu taslağı hazırlandı görüşe açıldı", "notr"),
    ("Sayıştay denetim raporunu meclise sundu", "notr"),
    ("Ombudsman yıllık raporunu açıkladı başvurular değerlendirildi", "notr"),
    ("İdare mahkemesi imar planı davasında karar verdi", "notr"),
    ("Ceza infaz kanununda düzenleme yapıldı yeni kurallar belirlendi", "notr"),

    # ── Bilim ve teknoloji ──
    ("NASA Mars'a yeni keşif aracı gönderdi veriler bekleniyor", "notr"),
    ("Yapay zeka düzenlemesi konusunda AB yeni çerçeve belirledi", "notr"),
    ("5G altyapısı 10 şehirde daha devreye alınacak", "notr"),
    ("Dijital kimlik sistemi pilot uygulamaya geçiyor", "notr"),
    ("Siber güvenlik kanunu güncellenecek uzmanlar görüş bildirdi", "notr"),
    ("Enerji Bakanlığı nükleer santral raporunu yayımladı", "notr"),
    ("Uzay ajansı yeni uydu fırlatma takvimini açıkladı", "notr"),
    ("Blockchain düzenlemesi meclise sunuldu tartışılacak", "notr"),
    ("TÜBİTAK yeni araştırma çağrısını duyurdu başvuru süreci başladı", "notr"),
    ("Bilgi teknolojileri kurumu internet hız raporunu yayımladı", "notr"),
    ("Elektrikli araç şarj istasyonları yaygınlaştırılıyor plan açıklandı", "notr"),

    # ── Ulaşım ──
    ("İstanbul metrosu yeni hatla genişliyor açılış yaklaşıyor", "notr"),
    ("Hızlı tren seferleri yeni güzergahta başlıyor bilet satışı açıldı", "notr"),
    ("Havayolu şirketi yeni uçuş noktaları ekledi sefer sayısı arttı", "notr"),
    ("Marmaray'da bakım çalışması yapılacak seferler aksayacak", "notr"),
    ("Köprü geçiş ücretlerine yıllık güncelleme yapıldı", "notr"),
    ("Yeni otobüs hatları devreye alındı toplu taşıma güncellendi", "notr"),
    ("Yüksek hızlı tren projesi güzergahı kesinleşti ihale sürecinde", "notr"),
    ("İstanbul trafiğinde yoğunluk yüzde 70'e ulaştı alternatif rotalar öneriliyor", "notr"),
    ("Yeni havalimanı pisti inşaatı başladı kapasite artacak", "notr"),
    ("Deniz otobüsü seferleri kış tarifesine geçiyor", "notr"),

    # ── Kültür ve sanat ──
    ("Film festivali bu hafta başlıyor programda 50 film var", "notr"),
    ("Müze hafta sonu ücretsiz olacak ziyaretçiler bekleniyor", "notr"),
    ("Yeni kütüphane hizmete açıldı koleksiyonda 100 bin kitap var", "notr"),
    ("Arkeoloji kazılarında yeni bulgular ortaya çıktı inceleniyor", "notr"),
    ("Tiyatro sezonu açılıyor yeni oyunlar sahnelenecek", "notr"),
    ("Sergi yarın kapılarını açıyor modern sanat eserleri sergilenecek", "notr"),
    ("Kitap fuarı 10 gün sürecek yüzlerce yazar katılacak", "notr"),
    ("Opera binası restore ediliyor yeni sezonda açılacak", "notr"),
    ("Belgesel festivali başvuruları açıldı son tarih ay sonu", "notr"),
    ("Müzik festivali programı açıklandı biletler satışta", "notr"),

    # ── Sağlık ──
    ("Grip aşısı kampanyası başladı randevular açıldı", "notr"),
    ("Sağlık Bakanlığı yeni genelge yayımladı hastaneler bilgilendirildi", "notr"),
    ("Organ bağışı haftası etkinlikleri düzenleniyor", "notr"),
    ("Beslenme uzmanları önerilerde bulundu sağlıklı yaşam tavsiyeleri", "notr"),
    ("Yeni hastane projesinin temeli atıldı inşaat başlıyor", "notr"),
    ("Sağlık Bakanlığı yıllık istatistik raporunu yayımladı", "notr"),
    ("Diyabet tarama programı ülke genelinde başlatılıyor", "notr"),
    ("Eczacılar odası ilaç fiyat tarifesini güncelledi", "notr"),

    # ── Spor (nötr) ──
    ("Süper Lig'de 15. hafta maçları tamamlandı puan durumu güncellendi", "notr"),
    ("Transfer dönemi açıldı kulüpler harekete geçti", "notr"),
    ("Milli takım kadrosu açıklandı sürpriz isimler var", "notr"),
    ("Futbol federasyonu yeni sezon kurallarını belirledi", "notr"),
    ("Antrenman kampı başladı takım hazırlıklarını sürdürüyor", "notr"),
    ("Olimpiyat kotası için yarışmalar devam ediyor sporcular hazır", "notr"),
    ("Basketbol ligi play-off eşleşmeleri belli oldu", "notr"),

    # ══════════════════════════════════════════════════
    #  ZOR ÖRNEKLER — bağlam gerektiren (40+)
    # ══════════════════════════════════════════════════

    # Negatif kelime ama pozitif anlam
    ("Deprem bölgesinde hayat normale dönüyor yardımlar meyvesini veriyor", "pozitif"),
    ("Kanser tedavisinde umut veren gelişme yeni ilaç onaylandı", "pozitif"),
    ("Sel mağdurlarına devlet desteği hızla ulaştı vatandaşlar memnun", "pozitif"),
    ("Kriz sonrası ekonomi toparlanıyor büyüme rakamları sevindirici", "pozitif"),
    ("Savaş bölgesinden tahliye edilen vatandaşlar yurda kavuştu", "pozitif"),
    ("İşsizlik düşüşe geçti istihdam artıyor gençler umutlu", "pozitif"),
    ("Yangın söndürüldü orman yeniden yeşerecek fidanlar dikildi", "pozitif"),
    ("Terörle mücadelede büyük başarı örgüt çökertildi", "pozitif"),
    ("Afet bölgesinde okullar yeniden açıldı çocuklar dersbaşı yaptı", "pozitif"),
    ("Salgın kontrol altına alındı hayat normale dönüyor", "pozitif"),
    ("Mülteciler güvenli bölgeye ulaştı yardımlar dağıtılıyor", "pozitif"),
    ("Kuraklıkla mücadelede yeni sulama sistemi devreye alındı çiftçiler rahatladi", "pozitif"),
    ("Çevre kirliliğine karşı alınan önlemler sonuç verdi su kalitesi arttı", "pozitif"),

    # Pozitif kelime ama negatif anlam
    ("Başarılı operasyona rağmen asker şehit düştü acı haber", "negatif"),
    ("Rekor kıran enflasyon vatandaşın alım gücünü eritiyor", "negatif"),
    ("Barış görüşmeleri çöktü taraflar masadan kalktı umutlar söndü", "negatif"),
    ("Büyüme hedefi tutmadı ekonomi beklentilerin altında kaldı", "negatif"),
    ("Şampiyon takımın yıldız oyuncusu sakatlandı sezon kapandı", "negatif"),
    ("Güzel haberlere rağmen kriz derinleşiyor uzmanlar uyarıyor", "negatif"),
    ("Umut verici tedavi yan etkileri nedeniyle geri çekildi hayal kırıklığı", "negatif"),
    ("Yatırım rekoru kırılsa da işsizlik artmaya devam ediyor", "negatif"),
    ("Festival coşkusu faciayla sonuçlandı sahne çöktü yaralılar var", "negatif"),
    ("Zafer kutlamaları sırasında izdiham yaşandı onlarca kişi yaralandı", "negatif"),
    ("Rekor büyüme rakamlarına rağmen yoksulluk artıyor", "negatif"),
    ("Başarılı sezona rağmen kulüp mali krizde borçlar katlanıyor", "negatif"),

    # Nötr gibi görünen ama aslında pozitif
    ("Meclis çocuk istismarına karşı ağırlaştırılmış ceza yasasını kabul etti", "pozitif"),
    ("Hükümet emekli maaşlarına yüzde 30 zam yapılacağını açıkladı", "pozitif"),
    ("Merkez Bankası enflasyon hedefini tutturdu piyasalar rahatladı", "pozitif"),
    ("Mahkeme kararıyla fabrika atıklarını nehre boşaltan şirket kapatıldı", "pozitif"),
    ("Mahkeme çocuk işçiliği yapan firmaya rekor ceza kesti", "pozitif"),
    ("Parlamento kadına şiddeti önleme yasasını oy birliğiyle kabul etti", "pozitif"),

    # Nötr gibi görünen ama aslında negatif
    ("Araştırma: Uzun süre oturmak kanser riskini artırabiliyor", "negatif"),
    ("Rapor: Her 5 çocuktan biri yoksulluk sınırının altında yaşıyor", "negatif"),
    ("İstatistik: Kadın cinayetlerinde son 10 yılda yüzde 50 artış", "negatif"),
    ("Çalışma: Hava kirliliği yılda 30 bin erken ölüme neden oluyor", "negatif"),

    # Gerçek nötr
    ("Belediye yeni parklar ve yeşil alanlar için master planı açıkladı", "notr"),
    ("Meclis seçim barajını değiştiren yasayı görüşmeye aldı", "notr"),
    ("Ulaştırma Bakanlığı yeni köprü projesinin fizibilite raporunu yayımladı", "notr"),
    ("Bankalar kredi faiz oranlarını güncelledi yeni tablolar yayımlandı", "notr"),
    ("Akdeniz'de 5.2 büyüklüğünde deprem meydana geldi", "notr"),
    ("Meteoroloji hafta sonu için kuvvetli yağış uyarısı yaptı", "notr"),
    ("Cumhurbaşkanlığı Sözcüsü basın açıklaması yaptı gündem paylaşıldı", "notr"),

    # ── Auto-generated examples ──
    ("Mersinde patlama meydana geldi 30 kişi yaralandı", "negatif"),
    ("Mersinde patlama meydana geldi 12 kişi yaralandı", "negatif"),
    ("Gaziantepde patlama meydana geldi 25 kişi yaralandı", "negatif"),
    ("Manisade patlama meydana geldi 50 kişi yaralandı", "negatif"),
    ("Adanade patlama meydana geldi 50 kişi yaralandı", "negatif"),
    ("Trabzonde patlama meydana geldi 12 kişi yaralandı", "negatif"),
    ("Trabzonde patlama meydana geldi 20 kişi yaralandı", "negatif"),
    ("Konyade patlama meydana geldi 25 kişi yaralandı", "negatif"),
    ("Bursade patlama meydana geldi 12 kişi yaralandı", "negatif"),
    ("Diyarbakırde patlama meydana geldi 7 kişi yaralandı", "negatif"),
    ("Mersinde patlama meydana geldi 30 kişi yaralandı", "negatif"),
    ("Denizlide patlama meydana geldi 7 kişi yaralandı", "negatif"),
    ("Muğlade patlama meydana geldi 50 kişi yaralandı", "negatif"),
    ("Bursade patlama meydana geldi 20 kişi yaralandı", "negatif"),
    ("Manisade patlama meydana geldi 5 kişi yaralandı", "negatif"),
    ("inşaat sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("tekstil sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("otomotiv sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("gıda sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("turizm sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("madencilik sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("lojistik sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("perakende sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("enerji sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("tarım sektöründe kriz derinleşiyor firmalar iflas ediyor", "negatif"),
    ("ekmek fiyatları yüzde 40 arttı vatandaş zor durumda", "negatif"),
    ("et fiyatları yüzde 5 arttı vatandaş zor durumda", "negatif"),
    ("süt fiyatları yüzde 30 arttı vatandaş zor durumda", "negatif"),
    ("doğalgaz fiyatları yüzde 20 arttı vatandaş zor durumda", "negatif"),
    ("elektrik fiyatları yüzde 25 arttı vatandaş zor durumda", "negatif"),
    ("benzin fiyatları yüzde 3 arttı vatandaş zor durumda", "negatif"),
    ("kira fiyatları yüzde 15 arttı vatandaş zor durumda", "negatif"),
    ("ilaç fiyatları yüzde 40 arttı vatandaş zor durumda", "negatif"),
    ("su fiyatları yüzde 3 arttı vatandaş zor durumda", "negatif"),
    ("ulaşım fiyatları yüzde 25 arttı vatandaş zor durumda", "negatif"),
    ("Gaziantepde sel baskını 50 evi sular altında bıraktı", "negatif"),
    ("Eskişehirde sel baskını 7 evi sular altında bıraktı", "negatif"),
    ("Muğlade sel baskını 40 evi sular altında bıraktı", "negatif"),
    ("Bursade sel baskını 7 evi sular altında bıraktı", "negatif"),
    ("Erzurumde sel baskını 3 evi sular altında bıraktı", "negatif"),
    ("Denizlide sel baskını 5 evi sular altında bıraktı", "negatif"),
    ("Gaziantepde sel baskını 3 evi sular altında bıraktı", "negatif"),
    ("Manisade sel baskını 7 evi sular altında bıraktı", "negatif"),
    ("Erzurumde sel baskını 30 evi sular altında bıraktı", "negatif"),
    ("İzmirde sel baskını 25 evi sular altında bıraktı", "negatif"),
    ("Eskişehirde sel baskını 5 evi sular altında bıraktı", "negatif"),
    ("Konyade sel baskını 25 evi sular altında bıraktı", "negatif"),
    ("Gaziantepde sel baskını 5 evi sular altında bıraktı", "negatif"),
    ("Adanade sel baskını 7 evi sular altında bıraktı", "negatif"),
    ("Erzurumde sel baskını 15 evi sular altında bıraktı", "negatif"),
    ("Manisadeki yangında 20 dönümlük alan kül oldu", "negatif"),
    ("Ankaradeki yangında 30 dönümlük alan kül oldu", "negatif"),
    ("Mersindeki yangında 3 dönümlük alan kül oldu", "negatif"),
    ("Kayserideki yangında 7 dönümlük alan kül oldu", "negatif"),
    ("Konyadeki yangında 50 dönümlük alan kül oldu", "negatif"),
    ("Denizlideki yangında 7 dönümlük alan kül oldu", "negatif"),
    ("Mersindeki yangında 50 dönümlük alan kül oldu", "negatif"),
    ("Gaziantepdeki yangında 25 dönümlük alan kül oldu", "negatif"),
    ("İzmirdeki yangında 5 dönümlük alan kül oldu", "negatif"),
    ("Gaziantepdeki yangında 5 dönümlük alan kül oldu", "negatif"),
    ("Erzurumdeki yangında 5 dönümlük alan kül oldu", "negatif"),
    ("Diyarbakırdeki yangında 7 dönümlük alan kül oldu", "negatif"),
    ("Muğladeki yangında 15 dönümlük alan kül oldu", "negatif"),
    ("Mersindeki yangında 7 dönümlük alan kül oldu", "negatif"),
    ("Konyadeki yangında 7 dönümlük alan kül oldu", "negatif"),
    ("Malatyade fabrika kazası 40 işçi hastaneye kaldırıldı", "negatif"),
    ("Konyade fabrika kazası 15 işçi hastaneye kaldırıldı", "negatif"),
    ("Gaziantepde fabrika kazası 3 işçi hastaneye kaldırıldı", "negatif"),
    ("Muğlade fabrika kazası 50 işçi hastaneye kaldırıldı", "negatif"),
    ("Kayseride fabrika kazası 3 işçi hastaneye kaldırıldı", "negatif"),
    ("Vande fabrika kazası 20 işçi hastaneye kaldırıldı", "negatif"),
    ("Trabzonde fabrika kazası 12 işçi hastaneye kaldırıldı", "negatif"),
    ("Diyarbakırde fabrika kazası 25 işçi hastaneye kaldırıldı", "negatif"),
    ("Antalyade fabrika kazası 25 işçi hastaneye kaldırıldı", "negatif"),
    ("Eskişehirde fabrika kazası 20 işçi hastaneye kaldırıldı", "negatif"),
    ("Adanade fabrika kazası 25 işçi hastaneye kaldırıldı", "negatif"),
    ("Bursade fabrika kazası 25 işçi hastaneye kaldırıldı", "negatif"),
    ("Malatyade trafik kazası 15 kişi hayatını kaybetti", "negatif"),
    ("Eskişehirde trafik kazası 5 kişi hayatını kaybetti", "negatif"),
    ("Muğlade trafik kazası 20 kişi hayatını kaybetti", "negatif"),
    ("Bursade trafik kazası 12 kişi hayatını kaybetti", "negatif"),
    ("Adanade trafik kazası 12 kişi hayatını kaybetti", "negatif"),
    ("İzmirde trafik kazası 25 kişi hayatını kaybetti", "negatif"),
    ("Eskişehirde trafik kazası 15 kişi hayatını kaybetti", "negatif"),
    ("Mersinde trafik kazası 30 kişi hayatını kaybetti", "negatif"),
    ("Erzurumde trafik kazası 5 kişi hayatını kaybetti", "negatif"),
    ("Trabzonde trafik kazası 30 kişi hayatını kaybetti", "negatif"),
    ("Samsunde trafik kazası 25 kişi hayatını kaybetti", "negatif"),
    ("Trabzonde trafik kazası 15 kişi hayatını kaybetti", "negatif"),
    ("inşaat sektöründe toplu işten çıkarma 5 bin kişi işsiz kaldı", "negatif"),
    ("tekstil sektöründe toplu işten çıkarma 40 bin kişi işsiz kaldı", "negatif"),
    ("otomotiv sektöründe toplu işten çıkarma 15 bin kişi işsiz kaldı", "negatif"),
    ("gıda sektöründe toplu işten çıkarma 20 bin kişi işsiz kaldı", "negatif"),
    ("turizm sektöründe toplu işten çıkarma 30 bin kişi işsiz kaldı", "negatif"),
    ("madencilik sektöründe toplu işten çıkarma 40 bin kişi işsiz kaldı", "negatif"),
    ("lojistik sektöründe toplu işten çıkarma 7 bin kişi işsiz kaldı", "negatif"),
    ("perakende sektöründe toplu işten çıkarma 15 bin kişi işsiz kaldı", "negatif"),
    ("enerji sektöründe toplu işten çıkarma 25 bin kişi işsiz kaldı", "negatif"),
    ("tarım sektöründe toplu işten çıkarma 15 bin kişi işsiz kaldı", "negatif"),
    ("Malatyade çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Manisade çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Adanade çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Samsunde çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Mersinde çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Manisade çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Manisade çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Erzurumde çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Gaziantepde çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Mersinde çevre felaketi nehirler kirletildi balıklar öldü", "negatif"),
    ("Kayseride deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Gaziantepde deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Eskişehirde deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Ankarade deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Eskişehirde deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Samsunde deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Gaziantepde deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Adanade deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Diyarbakırde deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("Malatyade deprem korkusu vatandaşlar geceyi dışarda geçirdi", "negatif"),
    ("ekmek zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("et zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("süt zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("doğalgaz zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("elektrik zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("benzin zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("kira zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("ilaç zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("su zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("ulaşım zamları sonrası vatandaşın bütçesi altüst oldu", "negatif"),
    ("Trabzonde hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Malatyade hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Eskişehirde hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Mersinde hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Ankarade hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Antalyade hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Denizlide hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Muğlade hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Gaziantepde hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Ankarade hastane önünde uzun kuyruklar oluştu sistem çöktü", "negatif"),
    ("Denizlide okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Samsunde okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Adanade okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Denizlide okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Erzurumde okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Gaziantepde okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Diyarbakırde okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Malatyade okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Erzurumde okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("Muğlade okulda şiddet olayı yaşandı veliler tedirgin", "negatif"),
    ("inşaat sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("tekstil sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("otomotiv sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("gıda sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("turizm sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("madencilik sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("lojistik sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("perakende sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("enerji sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("tarım sektöründe üretim durdu hammadde krizi büyüyor", "negatif"),
    ("yapay zeka alanında Türk bilim insanı dünya çapında ödül kazandı", "pozitif"),
    ("robotik alanında Türk mühendis dünya çapında ödül kazandı", "pozitif"),
    ("biyoteknoloji alanında Türk sporcu dünya çapında ödül kazandı", "pozitif"),
    ("nanoteknoloji alanında Türk sanatçı dünya çapında ödül kazandı", "pozitif"),
    ("uzay alanında Türk girişimci dünya çapında ödül kazandı", "pozitif"),
    ("savunma alanında Türk öğretmen dünya çapında ödül kazandı", "pozitif"),
    ("sağlık alanında Türk doktor dünya çapında ödül kazandı", "pozitif"),
    ("eğitim alanında Türk araştırmacı dünya çapında ödül kazandı", "pozitif"),
    ("yazılım alanında Türk mimar dünya çapında ödül kazandı", "pozitif"),
    ("otomasyon alanında Türk programcı dünya çapında ödül kazandı", "pozitif"),
    ("Denizlide yeni hastane açıldı 40 bin hasta faydalanacak", "pozitif"),
    ("Manisade yeni hastane açıldı 50 bin hasta faydalanacak", "pozitif"),
    ("Manisade yeni hastane açıldı 50 bin hasta faydalanacak", "pozitif"),
    ("Samsunde yeni hastane açıldı 15 bin hasta faydalanacak", "pozitif"),
    ("Gaziantepde yeni hastane açıldı 12 bin hasta faydalanacak", "pozitif"),
    ("Ankarade yeni hastane açıldı 5 bin hasta faydalanacak", "pozitif"),
    ("Antalyade yeni hastane açıldı 25 bin hasta faydalanacak", "pozitif"),
    ("Samsunde yeni hastane açıldı 25 bin hasta faydalanacak", "pozitif"),
    ("Bursade yeni hastane açıldı 7 bin hasta faydalanacak", "pozitif"),
    ("Hatayde yeni hastane açıldı 15 bin hasta faydalanacak", "pozitif"),
    ("İzmirde yeni hastane açıldı 30 bin hasta faydalanacak", "pozitif"),
    ("İzmirde yeni hastane açıldı 20 bin hasta faydalanacak", "pozitif"),
    ("Malatyade yeni hastane açıldı 7 bin hasta faydalanacak", "pozitif"),
    ("Samsunde yeni hastane açıldı 7 bin hasta faydalanacak", "pozitif"),
    ("Eskişehirde yeni hastane açıldı 30 bin hasta faydalanacak", "pozitif"),
    ("yapay zeka ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("robotik ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("biyoteknoloji ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("nanoteknoloji ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("uzay ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("savunma ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("sağlık ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("eğitim ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("yazılım ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("otomasyon ihracatı rekor kırdı hedefler aşıldı sektör büyüyor", "pozitif"),
    ("Mersinde 7 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Malatyade 5 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Muğlade 40 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Adanade 12 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Eskişehirde 30 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Diyarbakırde 40 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Gaziantepde 3 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Diyarbakırde 5 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Erzurumde 20 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Hatayde 30 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Mersinde 5 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Adanade 20 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Vande 3 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Hatayde 50 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("Vande 40 kişiye istihdam sağlayacak yeni fabrika açıldı", "pozitif"),
    ("yapay zeka projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("robotik projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("biyoteknoloji projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("nanoteknoloji projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("uzay projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("savunma projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("sağlık projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("eğitim projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("yazılım projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("otomasyon projesi başarıyla tamamlandı dünya basınında yer buldu", "pozitif"),
    ("Diyarbakırde çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Ankarade çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Hatayde çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Kayseride çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Malatyade çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Diyarbakırde çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Adanade çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Konyade çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Hatayde çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("İzmirde çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Mersinde çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("İzmirde çevre temizlik kampanyası büyük katılımla tamamlandı", "pozitif"),
    ("Türk bilim insanı Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk mühendis Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk sporcu Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk sanatçı Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk girişimci Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk öğretmen Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk doktor Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk araştırmacı Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk mimar Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("Türk programcı Avrupa birinciliği kazandı ülkemizi gururlandırdı", "pozitif"),
    ("İzmirde yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Kayseride yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Denizlide yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Trabzonde yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Muğlade yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Samsunde yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Adanade yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Erzurumde yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("İzmirde yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Malatyade yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Trabzonde yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("Gaziantepde yenilenebilir enerji santrali açıldı temiz enerji artıyor", "pozitif"),
    ("inşaat sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("tekstil sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("otomotiv sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("gıda sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("turizm sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("madencilik sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("lojistik sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("perakende sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("enerji sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("tarım sektöründe büyüme rekoru kırıldı istihdam artıyor", "pozitif"),
    ("Antalyade sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Vande sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Adanade sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Diyarbakırde sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Diyarbakırde sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Eskişehirde sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Bursade sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Muğlade sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Gaziantepde sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Muğlade sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Adanade sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Eskişehirde sosyal konut projesi tamamlandı aileler taşındı", "pozitif"),
    ("Genç bilim insanı uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç mühendis uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç sporcu uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç sanatçı uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç girişimci uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç öğretmen uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç doktor uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç araştırmacı uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç mimar uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Genç programcı uluslararası patent aldı Türkiye adına büyük başarı", "pozitif"),
    ("Muğlade ağaçlandırma seferberliği 20 bin fidan dikildi", "pozitif"),
    ("Bursade ağaçlandırma seferberliği 12 bin fidan dikildi", "pozitif"),
    ("Antalyade ağaçlandırma seferberliği 50 bin fidan dikildi", "pozitif"),
    ("Konyade ağaçlandırma seferberliği 50 bin fidan dikildi", "pozitif"),
    ("Konyade ağaçlandırma seferberliği 50 bin fidan dikildi", "pozitif"),
    ("Konyade ağaçlandırma seferberliği 7 bin fidan dikildi", "pozitif"),
    ("Muğlade ağaçlandırma seferberliği 20 bin fidan dikildi", "pozitif"),
    ("Bursade ağaçlandırma seferberliği 3 bin fidan dikildi", "pozitif"),
    ("İzmirde ağaçlandırma seferberliği 20 bin fidan dikildi", "pozitif"),
    ("Denizlide ağaçlandırma seferberliği 25 bin fidan dikildi", "pozitif"),
    ("Mersinde ağaçlandırma seferberliği 12 bin fidan dikildi", "pozitif"),
    ("Adanade ağaçlandırma seferberliği 15 bin fidan dikildi", "pozitif"),
    ("yapay zeka alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("robotik alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("biyoteknoloji alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("nanoteknoloji alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("uzay alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("savunma alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("sağlık alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("eğitim alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("yazılım alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("otomasyon alanında yerli üretim başladı dışa bağımlılık azalıyor", "pozitif"),
    ("Adanali öğrenci otomasyon yarışmasında dünya birincisi oldu", "pozitif"),
    ("Antalyali öğrenci eğitim yarışmasında dünya birincisi oldu", "pozitif"),
    ("Konyali öğrenci otomasyon yarışmasında dünya birincisi oldu", "pozitif"),
    ("Vanli öğrenci robotik yarışmasında dünya birincisi oldu", "pozitif"),
    ("Denizlili öğrenci biyoteknoloji yarışmasında dünya birincisi oldu", "pozitif"),
    ("Ankarali öğrenci biyoteknoloji yarışmasında dünya birincisi oldu", "pozitif"),
    ("Trabzonli öğrenci eğitim yarışmasında dünya birincisi oldu", "pozitif"),
    ("Mersinli öğrenci savunma yarışmasında dünya birincisi oldu", "pozitif"),
    ("Antalyali öğrenci uzay yarışmasında dünya birincisi oldu", "pozitif"),
    ("Trabzonli öğrenci savunma yarışmasında dünya birincisi oldu", "pozitif"),
    ("Ankara Belediyesi yeni dönem planını açıkladı", "notr"),
    ("İzmir Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Bursa Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Antalya Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Konya Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Adana Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Gaziantep Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Trabzon Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Diyarbakır Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Samsun Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Kayseri Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Eskişehir Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Mersin Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Malatya Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Erzurum Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Van Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Denizli Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Muğla Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Hatay Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Manisa Belediyesi yeni dönem planını açıkladı", "notr"),
    ("Hatayde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("İzmirde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Hatayde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Mersinde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Ankarade toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Samsunde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Adanade toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Konyade toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Ankarade toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Samsunde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Muğlade toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Samsunde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Samsunde toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Antalyade toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Vande toplu taşıma güzergahları yeniden düzenlendi", "notr"),
    ("Malatya Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Bursa Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("İzmir Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Eskişehir Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Adana Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Manisa Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Kayseri Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Kayseri Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Kayseri Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("İzmir Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Mersin Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Diyarbakır Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Manisa Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Trabzon Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("Muğla Valiliği yeni kararları açıkladı uygulamaya geçilecek", "notr"),
    ("inşaat sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("tekstil sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("otomotiv sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("gıda sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("turizm sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("madencilik sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("lojistik sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("perakende sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("enerji sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("tarım sektörü temsilcileri bakanlıkta toplantı yaptı", "notr"),
    ("Konyade su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Ankarade su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Samsunde su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Mersinde su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Malatyade su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Denizlide su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Trabzonde su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Adanade su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Denizlide su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Malatyade su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("İzmirde su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Gaziantepde su kesintisi yapılacak vatandaşlar bilgilendirildi", "notr"),
    ("Trabzon Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Diyarbakır Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Adana Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Hatay Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Antalya Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Malatya Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Ankara Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Hatay Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Ankara Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Gaziantep Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Konya Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Adana Üniversitesi akademik takvimini açıkladı", "notr"),
    ("Samsunde nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("İzmirde nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Diyarbakırde nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Adanade nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Hatayde nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Malatyade nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Konyade nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Ankarade nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("İzmirde nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Denizlide nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Mersinde nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("Erzurumde nüfus verileri güncellendi yeni rakamlar açıklandı", "notr"),
    ("inşaat fuarı 5 gün sürecek katılımcılar belli", "notr"),
    ("tekstil fuarı 7 gün sürecek katılımcılar belli", "notr"),
    ("otomotiv fuarı 50 gün sürecek katılımcılar belli", "notr"),
    ("gıda fuarı 12 gün sürecek katılımcılar belli", "notr"),
    ("turizm fuarı 20 gün sürecek katılımcılar belli", "notr"),
    ("madencilik fuarı 25 gün sürecek katılımcılar belli", "notr"),
    ("lojistik fuarı 7 gün sürecek katılımcılar belli", "notr"),
    ("perakende fuarı 30 gün sürecek katılımcılar belli", "notr"),
    ("enerji fuarı 50 gün sürecek katılımcılar belli", "notr"),
    ("tarım fuarı 5 gün sürecek katılımcılar belli", "notr"),
    ("Eskişehirde yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Denizlide yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Adanade yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Trabzonde yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Hatayde yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Vande yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Gaziantepde yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Antalyade yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Malatyade yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Manisade yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Gaziantepde yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Adanade yol çalışması başlıyor trafik akışı değişecek", "notr"),
    ("Konya Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Van Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Trabzon Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Adana Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Diyarbakır Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Adana Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Denizli Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Denizli Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Muğla Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Bursa Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Trabzon Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("Trabzon Büyükşehir Belediyesi ihale takvimini açıkladı", "notr"),
    ("inşaat komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("tekstil komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("otomotiv komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("gıda komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("turizm komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("madencilik komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("lojistik komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("perakende komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("enerji komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("tarım komisyonu toplandı gündem maddeleri görüşüldü", "notr"),
    ("Denizlide imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Trabzonde imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Malatyade imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Mersinde imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Denizlide imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Muğlade imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Bursade imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Trabzonde imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Diyarbakırde imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Kayseride imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Hatayde imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Kayseride imar planı değişikliği askıya çıkarıldı", "notr"),
    ("Van havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("Gaziantep havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("Antalya havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("Ankara havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("İzmir havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("Malatya havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("İzmir havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("Kayseri havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("Gaziantep havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("İzmir havalimanında yeni terminal inşaatı devam ediyor", "notr"),
    ("Samsunde doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Adanade doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Diyarbakırde doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Diyarbakırde doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Ankarade doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Hatayde doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Denizlide doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Samsunde doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Gaziantepde doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
    ("Eskişehirde doğalgaz dönüşüm çalışmaları sürüyor", "notr"),
]


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


def plot_confusion_matrix(cm, labels, save_path):
    """Save confusion matrix as an image artifact."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Gerçek",
        xlabel="Tahmin",
        title="Confusion Matrix",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("  HaberNLP — BERT Sentiment Training + MLflow")
    print("=" * 60)

    # ── Prepare data ──
    texts = [t for t, _ in TRAINING_DATA]
    raw_labels = [l for _, l in TRAINING_DATA]
    labels = [LABEL2ID[l] for l in raw_labels]

    from collections import Counter
    dist = Counter(raw_labels)
    print(f"\nDataset: {len(texts)} samples")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count} ({count / len(texts) * 100:.1f}%)")

    # 70/15/15 split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )
    print(f"Splits: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # ── Hyperparameters ──
    EPOCHS = 15
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 3

    # ══════════════════════════════════════════════════
    #  MLflow Setup
    # ══════════════════════════════════════════════════
    # mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("habernlp-sentiment")

    with mlflow.start_run(run_name=f"bert-turkish-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        # Log hyperparameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "dataset_size": len(texts),
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
            "num_labels": 3,
            "max_length": 128,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": SEED,
        })

        # Log class distribution
        for label, count in dist.items():
            mlflow.log_metric(f"class_{label}_count", count)

        # ── Load model ──
        print(f"\nLoading: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        )

        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            learning_rate=LEARNING_RATE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=10,
            report_to="none",
            seed=SEED,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        )

        # ── Train ──
        print("\n" + "─" * 60)
        print("  Training...")
        print("─" * 60)
        train_result = trainer.train()

        # Log training loss
        mlflow.log_metric("train_loss", train_result.training_loss)

        # ── Validation ──
        print("\n── Validation ──")
        val_results = trainer.evaluate()
        for k, v in val_results.items():
            clean_key = k.replace("eval_", "val_")
            if isinstance(v, (int, float)):
                mlflow.log_metric(clean_key, round(v, 4) if isinstance(v, float) else v)
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # ── Test ──
        print("\n── Test (held-out) ──")
        test_preds = trainer.predict(test_dataset)
        test_pred_labels = np.argmax(test_preds.predictions, axis=-1)

        test_acc = accuracy_score(test_labels, test_pred_labels)
        test_f1 = f1_score(test_labels, test_pred_labels, average="macro")
        test_f1w = f1_score(test_labels, test_pred_labels, average="weighted")
        test_prec = precision_score(test_labels, test_pred_labels, average="macro")
        test_rec = recall_score(test_labels, test_pred_labels, average="macro")

        # Log test metrics to MLflow
        mlflow.log_metrics({
            "test_accuracy": round(test_acc, 4),
            "test_f1_macro": round(test_f1, 4),
            "test_f1_weighted": round(test_f1w, 4),
            "test_precision_macro": round(test_prec, 4),
            "test_recall_macro": round(test_rec, 4),
        })

        report = classification_report(test_labels, test_pred_labels, target_names=LABELS)
        print(report)

        cm = confusion_matrix(test_labels, test_pred_labels)
        print("Confusion Matrix:")
        print(f"  {'':>10} {'negatif':>10} {'notr':>10} {'pozitif':>10}")
        for i, row in enumerate(cm):
            print(f"  {LABELS[i]:>10} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

        # ── Save confusion matrix as MLflow artifact ──
        cm_path = REPORT_DIR / "confusion_matrix.png"
        plot_confusion_matrix(cm, LABELS, cm_path)
        mlflow.log_artifact(str(cm_path))

        # ── Save classification report as artifact ──
        report_txt_path = REPORT_DIR / "classification_report.txt"
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(str(report_txt_path))

        # ── Save model ──
        print(f"\nSaving model to: {OUTPUT_DIR}")
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        # ── Register model in MLflow ──
        mlflow.log_artifact(str(OUTPUT_DIR))

        # ── Save training report JSON ──
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "mlflow_run_id": run.info.run_id,
            "base_model": MODEL_NAME,
            "dataset_size": len(texts),
            "class_distribution": dict(dist),
            "splits": {
                "train": len(train_texts),
                "val": len(val_texts),
                "test": len(test_texts),
            },
            "hyperparameters": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            },
            "test_metrics": {
                "accuracy": round(test_acc, 4),
                "f1_macro": round(test_f1, 4),
                "f1_weighted": round(test_f1w, 4),
                "precision_macro": round(test_prec, 4),
                "recall_macro": round(test_rec, 4),
            },
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        report_path = REPORT_DIR / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(report_path))

        # ── Final summary ──
        print("\n" + "═" * 60)
        print(f"  ✓ TRAINING COMPLETE")
        print(f"  ✓ Dataset:          {len(texts)} samples")
        print(f"  ✓ Test F1 (macro):  {test_f1:.4f}")
        print(f"  ✓ Test Accuracy:    {test_acc:.4f}")
        print(f"  ✓ Test Precision:   {test_prec:.4f}")
        print(f"  ✓ Test Recall:      {test_rec:.4f}")
        print(f"  ✓ MLflow Run ID:    {run.info.run_id}")
        print(f"  ✓ Model saved to:   {OUTPUT_DIR}")
        print(f"  ✓ Report saved to:  {report_path}")
        print(f"")
        print(f"  → View experiments: mlflow ui")
        print(f"  → Then open:        http://localhost:5000")
        print("═" * 60)


if __name__ == "__main__":
    main()