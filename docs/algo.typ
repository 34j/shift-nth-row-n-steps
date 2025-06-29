#import "@preview/pavemat:0.2.0": pavemat
#set page(height: 240pt, width: 400pt, fill: black)
#set text(fill: white)


$
  pavemat(
  mat(a_(0 0), a_(0 1), a_(0 2); a_(1 0), a_(1 1), a_(1 2); a_(2 0), a_(2 1), a_(2 2))
  )
   &->_"pad"
   pavemat(
   mat(a_(0 0), a_(0 1), a_(0 2), 0, 0; a_(1 0), a_(1 1), a_(1 2), 0, 0; a_(2 0), a_(2 1), a_(2 2), 0, 0),
   pave: "dddDDSSSAAWWW"
   )
    \
  &->_"flatten"
  pavemat(
  mat(
    a_(0 0), a_(0 1), a_(0 2), 0, 0, a_(1 0), a_(1 1), a_(1 2), 0, 0, a_(2 0), a_(2 1), a_(2 2), 0, 0
  ),
  pave: "dddddSdddddWddSDDDWAAA"
  ) \
  &->_"slice []"
  pavemat(
  mat(a_(0 0), a_(0 1), a_(0 2), 0, 0, a_(1 0), a_(1 1), a_(1 2), 0, 0, a_(2 0), a_(2 1)),
  pave: "ddddSddddW"
  ) \
  &->_"reshape"
  pavemat(
  mat(a_(0 0), a_(0 1), a_(0 2), 0;0, a_(1 0), a_(1 1), a_(1 2); 0, 0, a_(2 0), a_(2 1)),
  pave: "dddDSSSAWWW") \
  &->_"(slice)" mat(a_(0 0), a_(0 1), a_(0 2); 0, a_(1 0), a_(1 1); 0, 0, a_(2 0))
$
