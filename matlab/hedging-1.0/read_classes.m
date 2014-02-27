function [classids, class_map] = read_classes(filename)

class_map = java.util.Hashtable;

fd = fopen(filename);

c = textscan(fd,'%s');

fclose(fd);

n = numel(c{1});

classids = c{1};

for i =1:n
    class_map.put(c{1}{i}, i);
end

