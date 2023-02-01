clearvars
clc
close all

X_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\X\';
X_test_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\X_test\';
Y_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\Y\';
Yc_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\Y_corrected\';
Yai_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\Y_ai\';
Yai_prel_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\Y_ai_prel\';
X_file_root='X_';
Xt_file_root='X_';
Y_file_root='Y_';
Yc_file_root='Y_';
Yai_file_root='Y_';

%errors=zeros(1,1000);
k=0;
figure
for i=0:1:99
    A_real=imread([X_folder X_file_root num2str(i) '.png']);
    A_noise=imread([Y_folder Y_file_root num2str(i) '.png']);
    A_hc=imread([Yc_folder Yc_file_root num2str(i) '_mod.png']);
    A_ai=imread([Yai_folder Yai_file_root num2str(i) '_ai.png']);
    A_test=imread([X_test_folder Xt_file_root num2str(i) '_test.png']);
    %A_ai_bw=imbinarize(rgb2gray(A_ai));
    % A_gray=A(:,:,1)*0.3a+A(:,:,2)*0.59+A(:,:,3)*0.11;
    
    A_test_gray=rgb2gray(A_test);
    It_filt = imfilter(A_test_gray,fspecial('gaussian',[5,5],0.9));
    IIt=It_filt;
    IIt(IIt>100)=255;
    IIt(IIt<=100)=0;
    
    file_name=[X_test_folder Xt_file_root num2str(i) '_test.png'];
    imwrite(IIt,[file_name(1:end-4) '_prel.png'],'PNG')
    
    A_gray=rgb2gray(A_ai);
    Ifilt = imfilter(A_gray,fspecial('gaussian',[5,5],0.9));
    II=Ifilt;
    II(II>100)=255;
    II(II<=100)=0;
    k=k+1;
    
    subplot(5,4,k),imshow(A_real),title('original')
    subplot(5,4,k+4),imshow(A_noise),title('cu zgomot')
    subplot(5,4,k+8),imshow(A_hc),title('corectat de om')
    subplot(5,4,k+12),imshow(A_ai),title('autoencoder')
    subplot(5,4,k+16),imshow(II),title('autoencoder+prelucrare')

    if (k==4)
        %pause
        k=0;
    end

end