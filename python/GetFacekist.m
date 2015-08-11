clear all;
fea1=[];
ii = 1;
fileID = fopen('facedb/train.txt','w');
fileID2 = fopen('facedb/val.txt','w');
fileID3 = fopen('facedb/all.txt','w');

for k=1:32
    query1=['../../Project/faceClustering/alignedFaces/' num2str(k) '/*.JPG'];
    query2=['../../Project/faceClustering/alignedFaces/' num2str(k) '/*.jpg'];
    ax = [dir(query1); dir(query2)];
    len1 = length(ax);
    
    for i=1:len1
        path1 = ['/home/jess/Project/faceClustering/alignedFaces/' num2str(k) '/' ax(i).name];
        fprintf(fileID3,'%s\n', path1);
        if (mod(ii, 5)~=0)
            fprintf(fileID,'%s %d\n', path1, k-1);
        else
%             if (mod(ii, 2)==0)
                fprintf(fileID,'%s %d\n', path1, k-1);
                fprintf(fileID2,'%s %d\n', path1, k-1);
%             else
%                 fprintf(fileID3,'%s %d\n', path1, k-1);
%             end
        end
        ii=ii+1;
  
    end

end
 fclose(fileID); fclose(fileID2);
 fclose(fileID3);