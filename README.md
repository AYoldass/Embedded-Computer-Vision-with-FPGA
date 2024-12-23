# Embedded-Computer-Vision-with-FPGA


# YOLOv9 Deployment on Zedboard

Bu repository, YOLOv9 nesne algılama algoritmasının Avnet ZedBoard üzerinde çalıştırılması için gerekli dosyaları içermektedir. Aşağıda klasör ve dosyaların açıklamaları bulunmaktadır:

## Klasörler

### .elf-file
Bu klasör, ZedBoard üzerinde çalıştırılabilir ELF dosyalarını içerir. Bu dosyalar DPU (Deep Processing Unit) üzerinde çalışacak şekilde derlenmiştir.

### Archs
Bu klasör, farklı donanım mimarileri için kullanılan yapılandırma dosyalarını içerir. Donanım-tabanlı optimizasyonlar için kullanılabilir.

### ONNX_MODEL
Bu klasör, YOLOv9 modelinin ONNX formatındaki versiyonunu içerir. ONNX (Open Neural Network Exchange), modellerin farklı frameworkler arasında taşınabilmesini sağlar.

### cfg
Bu klasör, YOLOv9 modeline ait yapılandırma dosyalarını içerir. Yapılandırma dosyaları, modelin hiperparametrelerini ve ağ mimarisini tanımlar.

### compiled_models
Derlenmiş modellerin saklandığı klasördür. Bu modeller, ZedBoard üzerinde çalışacak şekilde optimize edilmiştir.

### dpu_design_ip
DPU tasarımı ve entegre devre (IP) bloklarına ait dosyaları içerir. Bu dosyalar, donanım hızlandırıcının yapılandırılması için gereklidir.

### saved_detection
Bu klasör, YOLOv9 tarafından yapılan nesne tespitlerinin sonuçlarının saklandığı yerdir. Çıktılar genellikle resim dosyası formatında saklanır.

### vga_pre_project
BASYS3 üzerinde VGA çıkışı için önceden hazırlanmış projelere ait dosyaları içerir.

### yoloModel
YOLOv9 model dosyalarını içerir. Bu dosyalar modelin çalıştırılması sırasında kullanılır.

## Dosyalar

### .DS_Store
MacOS işletim sisteminde klasörlerin özelliklerini saklamak için kullanılan bir sistem dosyasıdır. İşlevsel bir etkisi yoktur.

### .gitattributes
Bu dosya, Git tarafından dosya işleme ayarlarını tanımlamak için kullanılır. Örneğin, metin dosyalarının satır sonlarını ayarlamak için kullanılabilir.

### .gitignore
Bu dosya, Git tarafından izlenmeyecek dosya ve klasörleri tanımlamak için kullanılır. Örneğin, geçici dosyalar veya derleme çıktıları gibi.

### QuantizeModel.ipynb
Bu Jupyter Notebook dosyası, YOLOv9 modelinin nicemleme (quantization) işlemlerini gerçekleştirmek için kullanılır. Modelin, ZedBoard üzerindeki DPU'da daha verimli çalışması için boyut ve doğruluk optimizasyonu yapılır.

---

Bu dosyaları ve klasörleri kullanarak, ZedBoard üzerinde YOLOv9 modelini çalıştırabilir ve nesne algılama işlemlerini hızlandırabilirsiniz. 



![detect1](https://github.com/user-attachments/assets/ae75a7d5-2c8b-4180-ad5f-247bf5ca3bfc)
