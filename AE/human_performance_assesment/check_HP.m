clearvars
clc
close all

X_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\X\';
Y_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\Y\';
Yc_folder='C:\Users\Stefan\Desktop\image_correction\Human_performance_test\Y_corrected\';
X_file_root='X_';
Y_file_root='Y_';
Yc_file_root='Y_';

k=0;
figure

for i=0:99
    X=imread([X_folder X_file_root num2str(i) '.png']);
    Y=imread([Y_folder Y_file_root num2str(i) '.png']);
    Yc=imread([Yc_folder Yc_file_root num2str(i) '_mod.png']);

    k=k+1;
    subplot(3,4,k),imshow(X),title('original')
    subplot(3,4,k+4),imshow(Yc),title('corectat de om')
    subplot(3,4,k+8),imshow(Y),title('zgomot')
    
    if (k==4)
        pause
        k=0;
    end
end




% %Standard IPT Image
% I = imread('cameraman.tif');
% %Its edges
% E = edge(I,'canny');
% %Dilate the edges
% Ed = imdilate(E,strel('disk',2));
% %Filtered image
% Ifilt = imfilter(I,fspecial('gaussian'));
% %Use Ed as logical index into I to and replace with Ifilt
% I(Ed) = Ifilt(Ed);
