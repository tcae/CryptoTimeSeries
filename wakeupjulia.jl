println("wake up julia")
# for ix in (1:3); println(ix);end
# for ix in (3:-1:1); println(ix);end
a = [ix for ix in vcat(1:3,3:-1:1)]; println(a)
size(a, 1)
