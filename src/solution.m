%%% projekt ISS: Analýza vplyvu rúšky na reč 
%%% autor: Lukáš Tkáč (xtkacl00@stud.fit.vutbr.cz)

%%%%%%%%%%%%%%%%%%%%%%%% 3.ULOHA %%%%%%%%%%%%%%%%%%%%%%%% 
% nacitanie zvukoveho vzorku, rozdelenie na ramkce, ustrednenie,
% normalizacia na dynamicky rozsah [-1,1]

%nacitane zvukovej vzorky tonu (1.01 sec) bez rusky
[off,Fs] = audioread('part_maskoff_tone.wav');
off = off'; % potrebujeme riadkovy vektor
% ustrednenie signalu
off = off - mean(off);
% normalizacia do dynamickeho rozsahu [-1,1]
off = off / max(abs(off));

%nacitane zvukovej vzorky tonu (1.01 sec) s ruskou
[on,Fs] = audioread('part_maskon_tone.wav');
on = on'; % potrebujeme riadkovy vektor
% ustrednenie signalu
on = on - mean(on);
% normalizacia do dynamickeho rozsahu [-1,1]
on = on / max(abs(on));

% v tejto casti sa predpoklada dlzka nahravok rovnakej dlzky  

samples_per_ms = (length(off)/1010); %how much samples is in 1 ms

% Vypocet dlzky ramca vo vzorkach z 1.01 sec (1010ms) vzorku tonu hlasu
one_frame_length =(samples_per_ms*20); % count of samples for 20 ms

% Vypocet prekrytia ramca vo vzorkach z 1.01 sec vzorku tonu hlasu
overlap = (samples_per_ms*10); % count of samples for 10 ms

% Rozdelim si signal na ramce o dlzke 20ms(320 vzoriek),
% s prekrytim 10 ms(160 vzoriek) pre 1.01 sec vzorku tonu s ruskou a bez
% rusky. Data si ulozim do matice. Dokopy mi vyjde 100 ramcov

%vzorka tonu hlasu bez rusky 
[off_frames,NB_TR]=trame(off,one_frame_length,overlap);

%vzorka tonu hlasu s ruskou 
[on_frames,NB_TR]=trame(on,one_frame_length,overlap);

read_frame = 40; % vybrate cislo ramca na spracovanie n-ty ramec zo 100

x = (1:one_frame_length)/Fs; %pocet vzoriek na jeden ramec/vzorkovacia frekvencia
y1 = off_frames(:,read_frame); 
y2 = on_frames(:,read_frame);

figure % new figure
% vytvorenie grafu pre oba ramce spolocne(ruska, bez nej)
[hAx,hLine1,hLine2] = plotyy(x,y1,x,y2);

title([num2str(read_frame),'. ','frame of mask tones']);
xlabel('Time [s]');

% spravenie miesta medzi krajnymi bodmi grafu a hranou zobrazania grafu
set(hAx(1),'YLim',[-1 1]);
set(hAx(2),'YLim',[-1 1]);
set(hAx(1),'XLim',[-0.001 0.021]);
set(hAx(2),'XLim',[-0.001 0.021]);

ylabel(hAx(1),'The frame of mask off tone'); % left y-axis 
ylabel(hAx(2),'The frame of mask on tone'); % right y-axis

% vytvorenie grafu pre ramec bez rusky
figure
plot(x,y1);
axis([-0.001 0.021 -1 1]);
title([num2str(read_frame),'. ','frame of mask off tone']);
xlabel('Time [s]');
ylabel('y');

% vytvorenie grafu pre ramec s ruskou
figure
plot(x,y2);
axis([-0.001 0.021 -1 1]);
title([num2str(read_frame),'. ','frame of mask on tone']);
xlabel('Time [s]');
ylabel('y');

%%%%%%%%%%%%%%%%%%%%%%%% 4.ULOHA %%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Central clipping %%%%%

signal_clipped = zeros(one_frame_length,1); % alokujem si miesto 
maskoff = off_frames(:,read_frame); 
maskon = on_frames(:,read_frame);

signal_clipped = centralclipping(maskoff,one_frame_length);

figure
plot(x,signal_clipped);
axis([-0.001 0.021 -0.1 1.1]);
title(['Central clipping 70% for ', num2str(read_frame),'. ','frame of mask off tone']);
ylabel('y');
xlabel('Time [s]');

%%%%% AutoCorrelation (funkcia na konci tohoto suboru medzi funkciami)%%%%%
%%% zdroje(mierne modifikovane)
%%% 1) https://www.mathworks.com/matlabcentral/fileexchange/52178-calculate-autocorrelation-function
%%% 2) http://www.fit.vutbr.cz/~grezl/ZRE/lectures/05_pitch_en.pdf

autocorrelation = zeros(1,one_frame_length); %alokujem zdroje 
autocorrelation = doautocorr(signal_clipped);
bound = 32; % nastavenie prahu od vzorku vyssieho ako 32 (Fs/32 = 500Hz)
 
[freq,lag_index,lag] = fundfreq(autocorrelation, one_frame_length, bound, Fs);

%vykreslim graf s autokorelaciou s popiskami pre lag a prah
figure
plot(1:one_frame_length,autocorrelation);

% ciara pre prah
boundline = xline(bound,'-',{'Threshold'});
boundline.LineWidth = 1;

% ciara pre lag
hold on
stem(lag_index, lag, 'filled');
hold off

title(['Autocorrelation for ', num2str(read_frame),'. ','frame of mask off tone']);
legend('Autocorr coef','Threshold', 'Lag');
axis([-10 one_frame_length+10 -0.1 1.1])
ylabel('y');
xlabel('Samples');

f0_off = zeros(100,1);
f0_on = zeros(100,1);
%%%%% Zakladna frekvencia f0 pre vsetky ramce (100)%%%%%
%%% off_frames - ramce pre nahravku bez rusky
%%% on_frames - ramce pre nahravku s ruskou 
%%% 
for k = 1:100
    % pre ramce bez rusky 
    s_clipped_off = centralclipping(off_frames(:,k),one_frame_length);
    autocorrelate_off = doautocorr(s_clipped_off);
    [fr_off,lag_idx_off,lag_off] = fundfreq(autocorrelate_off, one_frame_length, bound, Fs);
    f0_off(k,1) = fr_off;
    
    s_clipped_off = zeros(1,one_frame_length);
    autocorrelate_off = zeros(1,one_frame_length);
    fr_off = 0;
    
    % pre ramce s ruskou
    s_clipped_on = centralclipping(on_frames(:,k),one_frame_length);
    autocorrelate_on = doautocorr(s_clipped_on);
    [fr_on,lag_idx_on,lag_on] = fundfreq(autocorrelate_on, one_frame_length, bound, Fs);
    f0_on(k,1) = fr_on;
    
    s_clipped_on = zeros(1,one_frame_length);
    autocorrelate_on = zeros(1,one_frame_length);
    fr_on = 0;
end

% rozptyly a stredne hodnoty pre zakladne tony f0 s ruskou a bez rusky
stred_off = mean(f0_off);
stred_on = mean(f0_on);
rozptyl_off = var(f0_off);
rozptyl_on = var(f0_on);

% vypis grafu tonu f0 pre ton s ruskou a bez nej
figure
plot(1:100,f0_off);

hold on
plot(1:100,f0_on);
hold off

ylabel('f0');
xlabel('Frames');
legend('mask off','mask on');
title(['Fundamental frequency of frames']);

%%%%%%%%%%%%%%%%%%%%%%%% 5.ULOHA %%%%%%%%%%%%%%%%%%%%%%%%

%%% aplikacia implementovaneho DFT
N = 1024;

dft_mask_off = zeros(N,100);
dft_mask_on = zeros(N,100);

log_dft_mask_off = zeros(N,100);
log_dft_mask_on = zeros(N,100);

off_spectr = zeros(N/2+1,100);
on_spectr = zeros(N/2+1,100);



for frame_num = 1:100
dft_mask_off(:, frame_num) = do_dft(off_frames(:,frame_num),N);
dft_mask_on(:, frame_num) = do_dft(on_frames(:,frame_num),N);

log_dft_mask_off(:, frame_num) = 10*log10(abs(dft_mask_off(:, frame_num)).^2);
log_dft_mask_on(:, frame_num) = 10*log10(abs(dft_mask_on(:, frame_num)).^2);
end

% je nam postacujuca polovica 
for i = 1:100
    off_spectr(1:513, i) = log_dft_mask_off(1:513, i);
    on_spectr(1:513, i) = log_dft_mask_on(1:513, i);
end

spectra_maskoff = zeros(51300,1);
spectra_maskon = zeros(51300,1);

%z matice s DFT a aplikovanim logaritmom si vytvorim riadkovy vektor pre
%data s ruskou i bez nej

spectra_maskoff = reshape(off_spectr,1,[]);
spectra_maskon = reshape(on_spectr,1,[]);

%vykreslenie logaritmickeho vykonoveho spektrogramu bez rusky
figure
spectrogram(spectra_maskoff, hamming(one_frame_length),floor(overlap), 1024, Fs, 'yaxis');
axis([0 1.01 0 8]);
view(0,90);
title(['Spectrogram without mask']);

%vykreslenie logaritmickeho vykonoveho spektrogramu s ruskou
figure
spectrogram(spectra_maskon, hamming(one_frame_length),floor(overlap), 1024, Fs, 'yaxis');
axis([0 1.01 0 8]);
view(0,90);
title(['Spectrogram with mask']);

%%%%%%%%%%%%%%%%%%%%%%%% 6.ULOHA %%%%%%%%%%%%%%%%%%%%%%%%

%frekvencna charakteristika DFTsruskou/DFTbezrusky = H(ejw)
% cize H(ejw) = Y(ejw)/X(ejw)
Hejw = rdivide(abs(dft_mask_on),abs(dft_mask_off));
Hejw = Hejw(1:1024,1:100); 
Hejw = 10*log10(abs(Hejw).^2)

%spriemerujem aby som ziskal jednu frekvencnu charakteristiku
powerfreq = mean(Hejw,2)
%transponujem na riadkovy vektor
powerfreq = powerfreq';

cutpowerfreq = powerfreq(1,1:513);

% vykreslim vykonove spektrum
yy = cutpowerfreq;
xx = (1:length(cutpowerfreq));
figure
plot(xx,yy);
axis ([-10 523 -16 15]);
title(['Power spectrum of mask']);


%%%%%%%%%%%%%%%%%%%%%%%% 7.ULOHA %%%%%%%%%%%%%%%%%%%%%%%%

IFFFFFFFFFT = do_idft(powerfreq,N);
CUTIFFFFFFFFFT = IFFFFFFFFFT(1,1:513);
yyy = CUTIFFFFFFFFFT;
xxx = (1:length(CUTIFFFFFFFFFT));
figure
plot(xxx,yyy);
axis ([-10 523 -0.8 0.8]);
title(['Impulse response']);


%%%%%%%%%%%%%%%%%%%%%%%% 8.ULOHA %%%%%%%%%%%%%%%%%%%%%%%%

[testtone,Fs] = audioread('maskoff_tone.wav');
testtone = testtone'; % potrebujeme riadkovy vektor
% ustrednenie signalu
testtone = testtone - mean(testtone);
% normalizacia do dynamickeho rozsahu [-1,1]
testtone = testtone / max(abs(testtone));

[testsentence,Fs] = audioread('maskoff_sentence.wav');
testsentence = testsentence'; % potrebujeme riadkovy vektor
% ustrednenie signalu
testsentence = testsentence - mean(testsentence);
% normalizacia do dynamickeho rozsahu [-1,1]
testsentence = testsentence / max(abs(testsentence));

[testsentenceon,Fs] = audioread('maskon_sentence.wav');
testsentenceon = testsentenceon'; % potrebujeme riadkovy vektor
% ustrednenie signalu
testsentenceon = testsentenceon - mean(testsentenceon);
% normalizacia do dynamickeho rozsahu [-1,1]
testsentenceon = testsentenceon / max(abs(testsentenceon));

% vyfiltrujem svojim filtrom nahrávku tónu
filtered_tone = filter(CUTIFFFFFFFFFT,[1],testtone);
filtered_tone = real(double(filtered_tone));
audiowrite('sim_maskon_tone.wav',filtered_tone,Fs);

% vyfiltrujem svojim filtrom nahrávku vety
filtered_sentence = filter(CUTIFFFFFFFFFT,[1],testsentence);
filtered_sentence = real(double(filtered_sentence));
audiowrite('sim_maskon_sentence.wav',filtered_sentence,Fs);

llA = length(testsentence);
llB = length(testsentenceon);
llC = length(filtered_sentence);

% graf pre vetu bez ruska
figure
AA = (1:llA)/Fs;
BB = testsentence(1,:);
plot(AA,BB);
ylabel('y');
xlabel('Time [s]');
title('Sentence without mask');

% graf pre vetu s ruskom
figure
CC = (1:llB)/Fs;
DD = testsentenceon(1,:);
plot(CC,DD);
ylabel('y');
xlabel('Time [s]');
title('Sentence with mask');

% graf pre vetu so simulovanym ruskom
figure
EE = (1:llC)/Fs;
FF = filtered_sentence(1,:);
plot(EE,FF);
ylabel('y');
xlabel('Time [s]');
title('Sentence with simulated mask');

%implementácia inverznej DFT
function idft = do_idft(input_data,N)  
    %alokujem si miesto pre vektor
    idft = zeros(1,N);
     
        % aplikacia dft na data s ruskou a bez nej 
        for n = 0 : N - 1  
            sum = 0;
            for k = 0 : N - 1 %aplikacia vzorca pre dft na vstupne data
                sum = sum + times(input_data(1,k + 1), exp(j*2*pi*k*n/N));
            end
            idft(1,n + 1) = times((1/N), sum); %zapisem po kazdej iteracii dopocitanu sumu 
        end
        %zapisem vysledok dft
        idft = idft(1,1:N);
end
%implementácia DFT
function dft = do_dft(input_frame,N)
     
    %alokujem si miesto pre vektory
    align = zeros(N,1);
    dft = zeros(N,1);
    %dft_frame_normalized = zeros(N,1);
    
    % vyplnime zarovnany vektor na N datami ramca
        for frame_val = input_frame(:,1)
            align(1:320,1) = frame_val;
        end
     
     % 1. x˜[n] = x[ mod N(n)]. periodizujeme
     % dft_frame_normalized(:,1) = mod(align(:,1), N);

        % aplikacia dft na data s ruskou a bez nej 
        for k = 0 : N - 1  
            sum = 0;
            for n = 0 : N - 1 %aplikacia vzorca pre dft na vstupne data
                sum = sum + times(align(n + 1,1), exp(-j*2*pi*k*n/N));
            end
            dft(k + 1,1) = sum; %zapisem po kazdej iteracii dopocitanu sumu 
        end
        %zapisem vysledok dft
        dft = dft(1:N,1);
end
% stanovenie zakladnej frekvencie pre ramec 
function [freq,lag_index,lag] = fundfreq(frame_correlated, frame_lenght, bound, fs)  
    X = (bound:frame_lenght);
    Y = frame_correlated;
    
    lag = max(frame_correlated(bound:frame_lenght,1)); % najdenie max hodnoty na y-osi po autokorelacii(lag)    
    lag_index = find(Y==lag); %najdenie korenspodujuceho indexu na x-ovej osi, vzorky jde bol najdeni lag
    lag_index = lag_index(lag_index > bound); %ponechavam hodnotu vacsiu ako prah, tie pre prahom zmazem 
    lag_index = lag_index(1); % ak by za prahom bol lag typu "nespicateho zubu" staci mi jeden index z toho
    
    freq = fs/lag_index; %vypocitam zakladnu frekvenciu f0
end
% prebrata funkcia zo studijnej etapy 
function [xtr,NB_TR]=trame(x,TRAME,R)

% call: [xtr,NB_TR]=trame(x,TRAME,R)
% framing of input signal into frames of length TRAME with overlap R.
% matrix xtr contains frames in its columns. 
% NB_TR is the number of generated frames 
%   (horizontal dimension of matrix xtr). 
% -------------------------------------------------------------------
% appel : [xtr,NB_TR]=trame(x,TRAME,R)
% structuration de la zone selectionnee x en NB_TR (calculé) trames
% de TRAME (choisi) points avec un recouvrement de R(choisi) points
% entre trames successives
%
% modif Thu Mar 28 15:19:14 CET 2002
% adding ALLOCATION of the matrix a-priori !

NB_TR=fix((length(x)-TRAME)/(TRAME-R)+1);
[l,c]=size(x);
if(l>c)
x=x';
end

% jan adds:
xtr = zeros(TRAME,NB_TR); 

if (NB_TR ~= 0)
for tr=1:NB_TR
	xtr(:,tr)= x(1+(tr-1)*(TRAME-R):tr*(TRAME-R)+R)';
end
else disp(' zone sélectionnée trop petite par rapport ŕ la trame demandée')
end
end
function R = doautocorr(A)
N = length(A);
R = zeros(N,1);
R(1) = sum(A.*A);
for m = 2:N
    B = circshift(A,-(m-1));
    B = B(1:(N-m+1));
    R(m) = sum(B.*A(1:(N-m+1)));
end
R = R/R(1); % normalizacia 
end
% central clipping 
function signal_clipped = centralclipping(frame_to_clipping, frame_lenght)
% 70 % maxima absolutnej hodnoty 
max_value = 0.7*max(abs(frame_to_clipping));

    for r = 1:frame_lenght
        if frame_to_clipping(r) > max_value 
           signal_clipped(r) = 1;
        elseif  frame_to_clipping(r) < -max_value
            signal_clipped(r) = -1;
        else
            signal_clipped(r) = 0;
        end 
    end
end

