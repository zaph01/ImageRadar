# 1. Neuronale Netze

Neuronale Netzwerke sind ein grundlegendes Konzept im Bereich des maschinellen Lernens und der künstlichen Intelligenz. Inspiriert von der Funktionsweise des menschlichen Gehirns bestehen sie aus vielen miteinander verbundenen künstlichen Neuronen, die in Schichten organisiert sind. <br>
Diese Neuronen verarbeiten Eingaben, führen Berechnungen durch und geben Ausgaben weiter, wobei sie durch Gewichtungen und Aktivierungsfunktionen modelliert werden. <br>
Durch Training mit großen Datensätzen können neuronale Netzwerke Muster und komplexe Zusammenhänge in den Daten erkennen und lernen, entsprechende Vorhersagen zu treffen oder Aufgaben zu erfüllen. Ihre Flexibilität und Anpassungsfähigkeit machen sie zu einer leistungsfähigen Technik in Bereichen wie Bilderkennung, Sprachverarbeitung, autonomen Fahrzeugen, medizinischer Diagnose und vielem mehr.

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image.png)

Eine spezielle Form der Neuronalen Netze sind die sogenannten Convolutional Neural Networks. <br>
Diese sind besonders gut für die Verarbeitung von Bild- und Sequenzdaten geeignet sind. Ihr Aufbau basiert auf dem Konzept von Faltungsschichten, die lokale Muster in den Eingabedaten erkennen und extrahieren.
![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-2.png)

# 2. Erläuterung des Papers
Grundlage dieser Arbeit ist ein Paper, sowie das zugehörige GitHub-Repository und ein Datensatz veröffentlicht von Valeo.ai
und erarbeitet von Julien Rebut, Arthur Ouaknine, Waqas Malik und Patrick Pérez. <br> 
Das Paper und das Repository sind unter "4. Anmerkungen" verlinkt.

## 2.1 Prämisse und Motivation des Papers

Die Prämisse des Papers besteht darin, die praktissche Anwendbarkeit von hochauflösenden Radarsystemen zu verbessern, insbesondere durch die Entwicklung einer optimierten Deep-Learning-Architektur und die Bereitstellung von Rohdaten für die Forschungsgemeinschaft. <br>
Die Motivation hinter der Arbeit liegt in der Herausforderung, die hohe Rechenkomplexität bei der Verarbeitung von HD-Radarbildern zu überwinden und die Effizienz von Radarsensoren für Fahrzeuganwendungen zu steigern. Dies wird angesichts der langsamen Fortschritte bei der Anwendung von Deep Learning für die Radarverarbeitung und des Mangels an öffentlich zugänglichen Datensätzen als besonders relevant erachtet.

## 2.2 Aufbau des Papers und des Modells

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-3.png)




### Datensatz
Der Datensatz des Papers "Raw High-Definition Radar for Multi-Task Learning" beinhaltet Daten von drei verschiedenen Sensortypen: Kamera, Radar und Laser (LIDAR). Er besteht aus 14025 einzelnen Fahrsituation mit den jeweils zugehörigen Sensordaten und Label.

### Modell 
Das Modell besteht aus fünf Blöcken, die im Folgenden jeweils kurz vorgestellt werden sollen.

#### MIMO pre-encoder
MIMO steht für "Multiple Inputs Multiple Outputs". Er transformiert die Range-Doppler-Spektren mittels deinterleaving in eine kompaktere Form. Diese Maßnahme wird ergriffen, da jedes detektierte Objekt im Range-Doppler-Spektrum mehrfach auftritt - einmal für jeden Sender. <br> 
Der MIMO pre-encoder hilft dabei, diese Signaturen zu organiseren und zu komprimieren.

#### FPN-Encoder
Der FPN-Encoder (Feature Pyramidial Network Encoder) dient dazu, die Merkmale aus den Radardaten zu extrahieren, und in verschiedene Maßstäben und Auflösungen zu erlernen. <br>
So lernt das Netz, Merkmale aus verschiedenen Skalen zu kombinieren, was das Netz insgesamt robuster und zuverlässiger gegenüber kontextsensitiven Information macht.

#### Range-Angle-Decoder 
Der Range-Angle Decoder passt die Achsreihenfolge der Eingabedaten an die Label an und skaliert die Eingabe auf eine höhere Auflösung. Des Weiteren werden Bereichsachse und Azimuthachse angeglichen, da erstere aufgrund voriger Verabeitung im Residual Block um den Faktor 2 reduziert wurde, während die Länge der Azimuthachse zunahm. <br>
Der Bereichsdecoder verwendet "Deconvolution-Schichten", um nur die Bereichsachse zu skalieren und Feature-Maps zu erzeugen, die mit denen aus der vorherigen Pyramidenstufe zusammengeführt werden. <br>
Ein abschließender Block von zwei Conv-BatchNorm-ReLU-Schichten erzeugt die endgültige Bereich-Azimut-Latenzrepräsentation.

# 3. Das Projekt ImageRader

## 3.1 Einleitung

In dem unserem Projekt zugrunde liegenden Paper werden die Rohdaten hochauflösender Radarsensoren in ein neuronales Netz gespeist. Dieses Netz soll den sogenannten "Free Driving Space", also den hindernisfreien Bereich der Fahrbahn vor dem Fahrzeug, berechnen. In dem unserem Projekt zugrunde liegenden Paper werden die Rohdaten hochauflösender Radarsensoren in ein neuronales Netz gespeist. Hierzu muss einerseits das Gebiet der Straße, andererseits auch die sich darauf befinden Hindernisse erkannt werden. <br>
Die Herausforderung darin besteht nicht nur in der Komplexität des Netzes, sondern auch in der dafür notwendigen enormen Rechenleistung, um schnell die richtigen Ergebnisse zu erhalten.

Die Motivation, dieses Projekt durchzuführen folgte aus der Frage: Ist es möglich ein Neuronales Netz gegen hochverarbeitete Daten zu trainieren? Und falls ja, ist es möglich, damit qualitativ ähnliche Ergebnisse zu erzielen? <br>
Diese Herangehensweise würde die erforderliche Rechenleistung stark reduzieren und eine Anwendung im Fahrzeug erleichtern.

Im Rahmen dieses Projektes ist es uns gelungen, einen Code für ein Neuronales Netz zu schreiben, der fehlerfrei kompiliert, jedoch noch nicht in Gänze zur Erkennung von Hindernissen auf der Straße in der Lage ist.
Er bildet eine gute Grundlage für kommende Projekte, die diesen Code nutzen können, um obige Fragen zu beantworten.

## 3.2 Grundlagenwissen Radar

### Radar

#### Allgemein

* Reichweite: 70m - 500m (je nach Auslegung der Antenne/Blickwinkel, sowie Zielgröße)
* Frequenzen:
    * Long-range: 76-77 GHz
    * Short-range: 77-81 GHz
* Für die interne Verarbeitung wird eine mittlere Datenrate (CAN) bis hohe Datenrate (100 Mbit Ethernet) benötigt
* Probleme: (feuchter) Schnee oder Eis vor dem Sensor
* 4D- Erkennung (Range, Azimuth, Elevation, Velocity)
    * Mehrere Antennen notwendig
* Die Signalstärke ist vom Objekt abhängig. Sowohl das Material, die Größe, sowie die Form spielen hierbei eine Rolle (Ein kleiner Tripelspiegel hat eine höhere Signalstärke als ein Haus).

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-4.png)

Zur Beurteilung wird der RCS- Wert herangenommen. Dieser gibt die effektive Reflexionsfläche eines Objekts an. 


* Je mehr Messzyklen (Frames) gemacht werden, desto höher ist die Genauigkeit der Messung, da mehr Messpunkte eines Objekts an den Antennen ankommen
    * Problem: Um frühe Reaktionen auszulösen, sind schnelle Messungen notwendig (Detektionszeit und Genauigkeit stehen in Relation)

#### Aufbau

* Transmitter (Tx) -> Ausstrahlen elektromagnetischer Wellen in bestimmte Richtung
* Radar Receiver (Rx) -> empfängt das reflektierte Signal

#### Verfahren

Zunächst wird eine oder mehrere Frequenzrampen (Chirp) ausgesendet. Das reflektierte Signal wird von den Receiver Antennen aufgenommen. Das empfangene Signal wird mit dem Sendesignal gemischt und mit einem Tiefpassfilter gefiltert. Es resultiert ein Signal mit der Differenzfrequenz aus Sende- und Empfangssignal. Diese Differenzfrequenz (Dopplerfrequenzverschiebung) ist proportional zur Entfernung des Objektes.

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-5.png)

#### Frequency Modulated Continous Wave Radar (FMCW Radar)

Hierbei handelt es sich um eine periodische Frequenzmodulation. Die Transmitter und Receiver arbeiten gleichzeitig.
Die Entfernungsmessung erfolgt über die Formel $𝑟= \frac{𝑐}{2} \frac{Δ𝑓}{𝑑𝑓/𝑑𝑡}$ mit c = Lichtgeschwindigkeit, $\frac{df}{dt}$ = Steilheit der Frequenzänderung, Δ𝑓= Frequenzänderung auf dem Weg r. <br>
Bei relativer Bewegung zum Radar, verschiebt sich das Empfangssignal um $f_D$ nach oben oder unten
![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-6.png)


* Mit FMCW sind sehr kleine Zielentfernungen möglich
* Entfernung und Radialgeschwindigkeit sind gleichzeitig messbar
* Signalverarbeitung in niedrigem Frequenzbereich -> einfache Schaltungstechnik

#### Winkelmessung

Für die Winkelmessung wird ein Antennen-Array benötigt. Die Auflösung verdoppelt sich mit jeder Sendeantenne.

Methoden:
* Unterschiedliche Empfindlichkeit in verschiedene Richtungen -> Winkelschätzung
* Ein ausgesendetes Signal kommt zu unterschiedlichen Zeiten am Receiver an -> Phasenunterschied $ΔΦ$
![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-7.png)
Der Phasenunterschied steigt linear zum Abstand der Antennen.

#### Radar Cube

Der Radar Cube ist eine dreidimensionale Datenstruktur.
Eine erste (Range-)FFT entlang jedem Chirp liefert die Entfernungsinformation.
Eine weitere (Doppler-)FFT liefert die Geschwindigkeitsinformation. <br>
Um den Cube zu erhalten, werden alle Rauminformationen aus allen Kanälen kombiniert.

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-8.png)

Ein Radar Cube lässt sich in eine Punktwolke umwandeln.

#### MIMO (Multiple Input Multiple Output)

Ein MIMO-Radar ist ein System mit vielen Antennen. N Sendeantennen und K Empfangsantennen ergeben ein rechnerisches virtuelles Feld von $N*K$ Elementen

Vorteile:
* Verbessertes räumliches Auflösungsvermögen
* Verbesserte Immunität gegen Störungen
* Verbesserte Entdeckungswahrscheinlichkeit der Ziele
* Da jeder Strahler einen eigenen Waveform-Generator hat, gibt es individuelle Signalformen

### Radarpunktwolken

Bei Punktwolken handelt es sich um eine 3D-Darstellung des Raums mit den empfangenen Signalen an den entsprechenden Positionen im Raum. Die Intensität des einzelnen Punktes gibt die Intensität des empfangenen Signals wieder. Kommen von einem Objekt mehrere Signale zurück so entsteht eine Ansammlung von Punkten (Cluster). Solche Cluster können mit Hilfe von neuronalen Netzen gruppiert und gelabelt werden.

### DDM - Doppler-Divison-Multiplexing

Bei einer DDM werden die Signale gleichzeitig ausgesendet. Die Sendeantennen haben dabei einen Phasenversatz $𝜔_k = \frac{2π(k-1)}{N}$ entlang der Geschwindigkeitsachse. Somit können die Signale in der Dopplerebene separiert werden.

Vorteil: Durch Zero-Padding kann eine genauere Geschwindigkeit des Objekts bestimmt werden

Nachteil: Die eindeutige Doppler- Geschwindigkeit ist um $\frac{1}{N}$ reduziert



## 3.3 Zugrundeliegende Daten

### Datensatz

Der gegebene “ready to use”-Datensatz von Valeo.ai enthält gelabelte Daten von verschiedenen Sensoren, die zum Trainieren eines neuronalen Netzes verwendet werden können. Um aussagekräftige Ergebnisse nach dem Trainieren zu erhalten, wird hierbei auf die Sensorfusion von Kamera, Radar und Lidar gesetzt. 
Ziel der Sensorfusion in der Entwicklung autonomer Fahrzeuge ist es, die Stärken und Schwächen der verschiedenen Sensoren auszugleichen. Zudem ermöglicht sie eine Redundanz für die Fahrerassistenzsysteme, um einen nächsten Schritt in Richtung des autonomen Fahrens zu machen.
    
Im Folgenden werden die zugrundeliegenden Daten beschrieben, die zum Training genutzt werden können.
    
- Kamera:
Kameradaten spielen eine entscheidende Rolle bei der Evaluierung von Daten, da sie eine eindeutige Objekterkennung und Klassifizierung ermöglichen.
Die Daten ermöglichen das Sammeln von visuellen Informationen, wie Details der Straße, Verkehrszeichen, Fahrzeuge usw.<br> 
In Bezug auf das maschinelle Lernen werden die Kamerabilder benutzt, um Muster zu erkennen und um Vorhersagen treffen zu können.

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-9.png)

- Laser-Punktwolke:
Eine Laser-Punktwolke ist eine Darstellung von dreidimensionalen Punkten im Raum. Sie enthält Informationen über die räumliche Verteilung von Objekten. Das Training eines neuronalen Netzwerkes auf Daten einer Laser-Punktwolke ermöglicht es, verschiedene Bereiche und Segmente interpretieren zu können. Dadurch können diese Daten genutzt werden, um Differenzierungen vorzunehmen, beispielsweise zwischen Straßenoberflächen und Gehwegen.

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-10.png)

- Radar-Punktwolke:
Sie stellt dreidimensionale Punkte im Raum dar. Neben den Möglichkeiten der Objekterkennung und Klassifizierung bekommt man durch das Training eines neuronalen Netzes auf Radar-Punktwolken Informationen zur Geschwindigkeit und Bewegung von Objekten. 

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-11.png)

- Radar FFT-Spektrum: 
Das Radar FFT-Spektrum leitet sich aus den Daten des Radars ab. In diesem Spektrum ist die Signalstärke im Frequenzbereich zu sehen. In Verbindung mit einem neuronalen Netz können FFT-Spektren dazu verwendet werden, um Objekte mittels charakteristischer Signaturen zu erkennen und zu verfolgen. Zudem kann man Informationen über die Bodenbeschaffenheit erhalten. Dies geschieht durch die unterschiedlichen Reflexionseigenschaften im Radar von den verschieden Untergründen (Asphalt, Schotter, usw.).

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-12.png)

- Segmentation Map:
Stellt den “Free-Driving Space” in einem Bild dar. Hierbei wird zwischen freien und besetzten Bereichen unterschieden. 

![show image](https://github.com/zaph01/ImageRadar/blob/main/ReadMe_files/image-13.png)

In unserem Programm haben wir ein neuronales Netzwerk entwickelt, das darauf trainiert wird, mithilfe von Radar-Punktwolken-Daten Objekte gegenüber den entsprechenden Labels zu erkennen.

### Label

Der gegebene Datensatz enthält eine dazugehörige labels.csv-Datei. In dieser Datei sind Informationen zu den Fahrsituationen enthalten. Gelabelt wurden die Fahrzeuge die pro Sample/Situation erkannt wurden.
Die csv-Datei enthält folgende Informationen:
- numSample:
Gibt die Nummer der aktuellen synchronisierten Situation zwischen allen Sensoren an. Diese
Information kann auf jeden einzelnen Sensor mit dem gemeinsamen dataset_index-Wert projiziert werden. Es können auch mehrere Zeilen die gleiche numSample-Nummer haben, es gibt eine Nummer pro Label.
- 2D-Begrenzungsrahmen der Fahrzeuge in Kamerakoordinaten:
Werte für [x1_pix, y1_pix, x2_pix, y2_pix], die ein begrenzendes Rechteck um das Fahrzeug bilden.
- 3D-Koordinaten des Fahrzeuges im Lidar-Koordinatensystem:
Werte für [laser_X_m, laser_Y_m, laser_Z_m], die die 3D-Position des Fahrzeuges angeben. Der angegebene Punkt liegt in der Mitte der Vorder- oder Rückseite des erkannten Fahrzeuges.
- 2D-Koordinaten des Fahrzeugs im Radar-Koordinatensystem:
Werte für [radar_X_m, radar_Y_m, radar_R_m (Range), radar_A_deg (Azimuth), radar_D (Doppler), radar_P_db (Leistung, des reflektierten Signals)], die die 2D-Koordinaten des Fahrzeuges (aus der Vogelperspektive) darstellen.
- zusätzliche Informationen:
    - dataset: Name der Sequenz, zu der das Fahrzeug gehört
    - dataset_index: Frame-Index in der aktuellen Frequenz
    - Difficult: 0 oder 1

## 3.4 Modell

### ImRadNet

In der Datei "ImRadNet.py" wird zunächst ein einfaches Neuronales Netz definiert, das über drei Convolutional Layer und drei Fully Connected Layer versucht eine grobe Erkennung von Hindernissen auf der Straße zu erkennen. <br>
Die Eingabedaten bestehen aus Radarpunktwolken mit Informationen über Entfernung, den Azimuth und die Dopplergeschwindigkeit.

### Training des Modells

Das Training des Modells wird in der Datei "1-Train.py" durchgeführt. Es werden 2500 Radarpunktwolken geladen, diese werden an das Neuronale Netz übergeben. Aus den Ausgaben des Neuronalen Netzes werden mittels einer auf unsere Anwendung leicht abgewandelten Form der Loss-Function aus der Arbeit on Valeo.ai die Abweichungen zwischen den Ausgaben des Neuronalen Netzes und der Labels berechnet. Diese Abweichungen bilden die Grundlage für die Anpassung der Trainingsparameter der nächsten Iteration. <br>
Zum Schluss wird das trainierte Netz für die weitere Verwendung gespeichert.

### Testen des Modells

In der Datei "2-Test.py" wird die Leistung des trainierten Neuronalen Netzes getestet. Insgesamt werden 500 Radarpunktwolken an das Netz übergeben und getestet, wie gut die Hindernisse auf der Straße erkannt werden.

# 4. Anmerkungen
Der Code zu diesem Projekt ist unter folgendem Link auf unserem GitHub Repository zu finden: https://github.com/zaph01/ImageRadar



Die Arbeit von Julien Rebut, Arthur Ouaknine, Waqas Malik und Patrick Pérez ist unter folgenden Links zu finden: 

Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Rebut_Raw_High-Definition_Radar_for_Multi-Task_Learning_CVPR_2022_paper.pdf

GitHub-Repository: https://github.com/valeoai/RADIal
