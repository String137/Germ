ач	
бЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Ъ

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
Р*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:Р*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Рс*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
Рс*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:с*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:с*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
х0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 0
value0B0 B0
Њ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
#_self_saveable_object_factories
	optimizer

signatures
	variables
regularization_losses
trainable_variables
	keras_api


kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
 	keras_api
w
#!_self_saveable_object_factories
"	variables
#regularization_losses
$trainable_variables
%	keras_api


&kernel
'bias
#(_self_saveable_object_factories
)	variables
*regularization_losses
+trainable_variables
,	keras_api
w
#-_self_saveable_object_factories
.	variables
/regularization_losses
0trainable_variables
1	keras_api
w
#2_self_saveable_object_factories
3	variables
4regularization_losses
5trainable_variables
6	keras_api
w
#7_self_saveable_object_factories
8	variables
9regularization_losses
:trainable_variables
;	keras_api


<kernel
=bias
#>_self_saveable_object_factories
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
w
#C_self_saveable_object_factories
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
w
#H_self_saveable_object_factories
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api


Mkernel
Nbias
#O_self_saveable_object_factories
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
w
#T_self_saveable_object_factories
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
w
#Y_self_saveable_object_factories
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
 
 
 
8
0
1
&2
'3
<4
=5
M6
N7
 
8
0
1
&2
'3
<4
=5
M6
N7
­
^layer_regularization_losses
_metrics
	variables
`layer_metrics
anon_trainable_variables
regularization_losses

blayers
trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
­
clayer_regularization_losses
dmetrics
elayer_metrics
	variables
fnon_trainable_variables
regularization_losses

glayers
trainable_variables
 
 
 
 
­
hlayer_regularization_losses
imetrics
jlayer_metrics
	variables
knon_trainable_variables
regularization_losses

llayers
trainable_variables
 
 
 
 
­
mlayer_regularization_losses
nmetrics
olayer_metrics
"	variables
pnon_trainable_variables
#regularization_losses

qlayers
$trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1
 

&0
'1
­
rlayer_regularization_losses
smetrics
tlayer_metrics
)	variables
unon_trainable_variables
*regularization_losses

vlayers
+trainable_variables
 
 
 
 
­
wlayer_regularization_losses
xmetrics
ylayer_metrics
.	variables
znon_trainable_variables
/regularization_losses

{layers
0trainable_variables
 
 
 
 
Ў
|layer_regularization_losses
}metrics
~layer_metrics
3	variables
non_trainable_variables
4regularization_losses
layers
5trainable_variables
 
 
 
 
В
 layer_regularization_losses
metrics
layer_metrics
8	variables
non_trainable_variables
9regularization_losses
layers
:trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1
 

<0
=1
В
 layer_regularization_losses
metrics
layer_metrics
?	variables
non_trainable_variables
@regularization_losses
layers
Atrainable_variables
 
 
 
 
В
 layer_regularization_losses
metrics
layer_metrics
D	variables
non_trainable_variables
Eregularization_losses
layers
Ftrainable_variables
 
 
 
 
В
 layer_regularization_losses
metrics
layer_metrics
I	variables
non_trainable_variables
Jregularization_losses
layers
Ktrainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1
 

M0
N1
В
 layer_regularization_losses
metrics
layer_metrics
P	variables
non_trainable_variables
Qregularization_losses
layers
Rtrainable_variables
 
 
 
 
В
 layer_regularization_losses
metrics
layer_metrics
U	variables
non_trainable_variables
Vregularization_losses
layers
Wtrainable_variables
 
 
 
 
В
 layer_regularization_losses
 metrics
Ёlayer_metrics
Z	variables
Ђnon_trainable_variables
[regularization_losses
Ѓlayers
\trainable_variables
 

Є0
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Ѕtotal

Іcount
Ї	variables
Ј	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ѕ0
І1

Ї	variables

serving_default_conv2d_2_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
Ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_2_inputconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_14536272
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
љ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_14536703
Ќ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_14536743О
ъ
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_14535951

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536589

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџР2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
т

*__inference_dense_3_layer_call_fn_14536623

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_145360572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
є
g
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_14536505

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
Ў
F__inference_conv2d_2_layer_call_and_return_conditional_losses_14536435

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х
f
G__inference_dropout_4_layer_call_and_return_conditional_losses_14535946

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_14536466

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_14536009

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:џџџџџџџџџР*
alpha%>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
П
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_14536543

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г
ц
/__inference_sequential_1_layer_call_fn_14536195
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_145361762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv2d_2_input
ж
­
E__inference_dense_2_layer_call_and_return_conditional_losses_14535988

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ2
Џ
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536140
conv2d_2_input
conv2d_2_14536110
conv2d_2_14536112
conv2d_3_14536117
conv2d_3_14536119
dense_2_14536125
dense_2_14536127
dense_3_14536132
dense_3_14536134
identityЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЊ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_14536110conv2d_2_14536112*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_145358362"
 conv2d_2/StatefulPartitionedCall
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_145358572
leaky_re_lu_4/PartitionedCall
dropout_3/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_145358822
dropout_3/PartitionedCallО
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_3_14536117conv2d_3_14536119*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_145359052"
 conv2d_3/StatefulPartitionedCall
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_145359262
leaky_re_lu_5/PartitionedCall
dropout_4/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_145359512
dropout_4/PartitionedCallі
flatten_1/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_145359702
flatten_1/PartitionedCallВ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_14536125dense_2_14536127*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_145359882!
dense_2/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_145360092
leaky_re_lu_6/PartitionedCallњ
dropout_5/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_145360342
dropout_5/PartitionedCallВ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_3_14536132dense_3_14536134*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_145360572!
dense_3/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_145360782
leaky_re_lu_7/PartitionedCallњ
reshape_1/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_145360982
reshape_1/PartitionedCall
IdentityIdentity"reshape_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv2d_2_input
Ю0
б
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536383

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpП
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_2/BiasAdd
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
leaky_re_lu_4/LeakyRelu
dropout_3/IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_3/IdentityА
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpд
conv2d_3/Conv2DConv2Ddropout_3/Identity:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_3/BiasAdd
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
leaky_re_lu_5/LeakyRelu
dropout_4/IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_4/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_4/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_1/ReshapeЇ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dense_2/MatMulЅ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02 
dense_2/BiasAdd/ReadVariableOpЂ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dense_2/BiasAdd
leaky_re_lu_6/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџР*
alpha%>2
leaky_re_lu_6/LeakyRelu
dropout_5/IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout_5/IdentityЇ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
Рс*
dtype02
dense_3/MatMul/ReadVariableOpЁ
dense_3/MatMulMatMuldropout_5/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
dense_3/MatMulЅ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:с*
dtype02 
dense_3/BiasAdd/ReadVariableOpЂ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
dense_3/BiasAdd
leaky_re_lu_7/LeakyRelu	LeakyReludense_3/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџс*
alpha%>2
leaky_re_lu_7/LeakyReluw
reshape_1/ShapeShape%leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicey
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :с2
reshape_1/Reshape/shape/1Ў
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape­
reshape_1/ReshapeReshape%leaky_re_lu_7/LeakyRelu:activations:0 reshape_1/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2
reshape_1/Reshapeo
IdentityIdentityreshape_1/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ:::::::::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
н

c
G__inference_reshape_1_layer_call_and_return_conditional_losses_14536645

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :с2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџс:P L
(
_output_shapes
:џџџџџџџџџс
 
_user_specified_nameinputs
Є
L
0__inference_leaky_re_lu_6_layer_call_fn_14536577

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_145360092
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
у!
Ў
!__inference__traced_save_14536703
file_prefix.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_218452ae791b4706887e2450d4092163/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameХ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*з
valueЭBЪB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesм
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*q
_input_shapes`
^: :::::
Р:Р:
Рс:с: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
Р:!

_output_shapes	
:Р:&"
 
_output_shapes
:
Рс:!

_output_shapes	
:с:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
Ф
e
,__inference_dropout_4_layer_call_fn_14536532

inputs
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_145359462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536034

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџР2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Х
f
G__inference_dropout_4_layer_call_and_return_conditional_losses_14536522

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
є
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_14536449

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
н
&__inference_signature_wrapper_14536272
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_145358222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv2d_2_input


+__inference_conv2d_3_layer_call_fn_14536500

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_145359052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
Ў
F__inference_conv2d_3_layer_call_and_return_conditional_losses_14536491

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и
g
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_14536628

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:џџџџџџџџџс*
alpha%>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџс:P L
(
_output_shapes
:џџџџџџџџџс
 
_user_specified_nameinputs
є
g
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_14535926

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
Ў
F__inference_conv2d_2_layer_call_and_return_conditional_losses_14535836

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
H
,__inference_dropout_4_layer_call_fn_14536537

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_145359512
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

H
,__inference_dropout_5_layer_call_fn_14536604

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_145360342
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
и
g
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_14536572

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:џџџџџџџџџР*
alpha%>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
И
H
,__inference_dropout_3_layer_call_fn_14536481

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_145358822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
н

c
G__inference_reshape_1_layer_call_and_return_conditional_losses_14536098

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :с2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџс:P L
(
_output_shapes
:џџџџџџџџџс
 
_user_specified_nameinputs
Л
о
/__inference_sequential_1_layer_call_fn_14536404

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_145361762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_14535970

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т

*__inference_dense_2_layer_call_fn_14536567

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_145359882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536029

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџР2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Х
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_14535877

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
о
/__inference_sequential_1_layer_call_fn_14536425

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_145362302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536594

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџР2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
и
g
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_14536078

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:џџџџџџџџџс*
alpha%>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџс:P L
(
_output_shapes
:џџџџџџџџџс
 
_user_specified_nameinputs
є
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_14535857

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ
H
,__inference_flatten_1_layer_call_fn_14536548

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_145359702
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
L
0__inference_leaky_re_lu_7_layer_call_fn_14536633

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_145360782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџс:P L
(
_output_shapes
:џџџџџџџџџс
 
_user_specified_nameinputs
Ј
e
,__inference_dropout_5_layer_call_fn_14536599

inputs
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_145360292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџР22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs

H
,__inference_reshape_1_layer_call_fn_14536650

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_145360982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџс:P L
(
_output_shapes
:џџџџџџџџџс
 
_user_specified_nameinputs
Р
L
0__inference_leaky_re_lu_4_layer_call_fn_14536454

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_145358572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І<

#__inference__wrapped_model_14535822
conv2d_2_input8
4sequential_1_conv2d_2_conv2d_readvariableop_resource9
5sequential_1_conv2d_2_biasadd_readvariableop_resource8
4sequential_1_conv2d_3_conv2d_readvariableop_resource9
5sequential_1_conv2d_3_biasadd_readvariableop_resource7
3sequential_1_dense_2_matmul_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource
identityз
+sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_2/Conv2D/ReadVariableOpю
sequential_1/conv2d_2/Conv2DConv2Dconv2d_2_input3sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
sequential_1/conv2d_2/Conv2DЮ
,sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_2/BiasAdd/ReadVariableOpр
sequential_1/conv2d_2/BiasAddBiasAdd%sequential_1/conv2d_2/Conv2D:output:04sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
sequential_1/conv2d_2/BiasAddТ
$sequential_1/leaky_re_lu_4/LeakyRelu	LeakyRelu&sequential_1/conv2d_2/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
alpha%>2&
$sequential_1/leaky_re_lu_4/LeakyReluМ
sequential_1/dropout_3/IdentityIdentity2sequential_1/leaky_re_lu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2!
sequential_1/dropout_3/Identityз
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_3/Conv2D/ReadVariableOp
sequential_1/conv2d_3/Conv2DConv2D(sequential_1/dropout_3/Identity:output:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
sequential_1/conv2d_3/Conv2DЮ
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpр
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
sequential_1/conv2d_3/BiasAddТ
$sequential_1/leaky_re_lu_5/LeakyRelu	LeakyRelu&sequential_1/conv2d_3/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
alpha%>2&
$sequential_1/leaky_re_lu_5/LeakyReluМ
sequential_1/dropout_4/IdentityIdentity2sequential_1/leaky_re_lu_5/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ2!
sequential_1/dropout_4/Identity
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
sequential_1/flatten_1/ConstЯ
sequential_1/flatten_1/ReshapeReshape(sequential_1/dropout_4/Identity:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
sequential_1/flatten_1/ReshapeЮ
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02,
*sequential_1/dense_2/MatMul/ReadVariableOpд
sequential_1/dense_2/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
sequential_1/dense_2/MatMulЬ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpж
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
sequential_1/dense_2/BiasAddК
$sequential_1/leaky_re_lu_6/LeakyRelu	LeakyRelu%sequential_1/dense_2/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџР*
alpha%>2&
$sequential_1/leaky_re_lu_6/LeakyReluЕ
sequential_1/dropout_5/IdentityIdentity2sequential_1/leaky_re_lu_6/LeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџР2!
sequential_1/dropout_5/IdentityЮ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
Рс*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOpе
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_5/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
sequential_1/dense_3/MatMulЬ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:с*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpж
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
sequential_1/dense_3/BiasAddК
$sequential_1/leaky_re_lu_7/LeakyRelu	LeakyRelu%sequential_1/dense_3/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџс*
alpha%>2&
$sequential_1/leaky_re_lu_7/LeakyRelu
sequential_1/reshape_1/ShapeShape2sequential_1/leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape_1/ShapeЂ
*sequential_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_1/reshape_1/strided_slice/stackІ
,sequential_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_1/strided_slice/stack_1І
,sequential_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_1/strided_slice/stack_2ь
$sequential_1/reshape_1/strided_sliceStridedSlice%sequential_1/reshape_1/Shape:output:03sequential_1/reshape_1/strided_slice/stack:output:05sequential_1/reshape_1/strided_slice/stack_1:output:05sequential_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_1/reshape_1/strided_slice
&sequential_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :с2(
&sequential_1/reshape_1/Reshape/shape/1т
$sequential_1/reshape_1/Reshape/shapePack-sequential_1/reshape_1/strided_slice:output:0/sequential_1/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/reshape_1/Reshape/shapeс
sequential_1/reshape_1/ReshapeReshape2sequential_1/leaky_re_lu_7/LeakyRelu:activations:0-sequential_1/reshape_1/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2 
sequential_1/reshape_1/Reshape|
IdentityIdentity'sequential_1/reshape_1/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ:::::::::_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv2d_2_input
ъ
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_14535882

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
e
,__inference_dropout_3_layer_call_fn_14536476

inputs
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_145358772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_14536471

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
7

J__inference_sequential_1_layer_call_and_return_conditional_losses_14536107
conv2d_2_input
conv2d_2_14535847
conv2d_2_14535849
conv2d_3_14535916
conv2d_3_14535918
dense_2_14535999
dense_2_14536001
dense_3_14536068
dense_3_14536070
identityЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallЊ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_14535847conv2d_2_14535849*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_145358362"
 conv2d_2/StatefulPartitionedCall
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_145358572
leaky_re_lu_4/PartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_145358772#
!dropout_3/StatefulPartitionedCallЦ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_3_14535916conv2d_3_14535918*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_145359052"
 conv2d_3/StatefulPartitionedCall
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_145359262
leaky_re_lu_5/PartitionedCallН
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_145359462#
!dropout_4/StatefulPartitionedCallў
flatten_1/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_145359702
flatten_1/PartitionedCallВ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_14535999dense_2_14536001*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_145359882!
dense_2/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_145360092
leaky_re_lu_6/PartitionedCallЖ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_145360292#
!dropout_5/StatefulPartitionedCallК
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_3_14536068dense_3_14536070*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_145360572!
dense_3/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_145360782
leaky_re_lu_7/PartitionedCallњ
reshape_1/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_145360982
reshape_1/PartitionedCallэ
IdentityIdentity"reshape_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv2d_2_input


+__inference_conv2d_2_layer_call_fn_14536444

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_145358362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р
L
0__inference_leaky_re_lu_5_layer_call_fn_14536510

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_145359262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
­
E__inference_dense_3_layer_call_and_return_conditional_losses_14536614

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Рс*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:с*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
щ6

J__inference_sequential_1_layer_call_and_return_conditional_losses_14536176

inputs
conv2d_2_14536146
conv2d_2_14536148
conv2d_3_14536153
conv2d_3_14536155
dense_2_14536161
dense_2_14536163
dense_3_14536168
dense_3_14536170
identityЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallЂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_14536146conv2d_2_14536148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_145358362"
 conv2d_2/StatefulPartitionedCall
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_145358572
leaky_re_lu_4/PartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_145358772#
!dropout_3/StatefulPartitionedCallЦ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_3_14536153conv2d_3_14536155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_145359052"
 conv2d_3/StatefulPartitionedCall
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_145359262
leaky_re_lu_5/PartitionedCallН
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_145359462#
!dropout_4/StatefulPartitionedCallў
flatten_1/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_145359702
flatten_1/PartitionedCallВ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_14536161dense_2_14536163*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_145359882!
dense_2/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_145360092
leaky_re_lu_6/PartitionedCallЖ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_145360292#
!dropout_5/StatefulPartitionedCallК
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_3_14536168dense_3_14536170*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_145360572!
dense_3/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_145360782
leaky_re_lu_7/PartitionedCallњ
reshape_1/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_145360982
reshape_1/PartitionedCallэ
IdentityIdentity"reshape_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
2
Ї
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536230

inputs
conv2d_2_14536200
conv2d_2_14536202
conv2d_3_14536207
conv2d_3_14536209
dense_2_14536215
dense_2_14536217
dense_3_14536222
dense_3_14536224
identityЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_14536200conv2d_2_14536202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_145358362"
 conv2d_2/StatefulPartitionedCall
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_145358572
leaky_re_lu_4/PartitionedCall
dropout_3/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_145358822
dropout_3/PartitionedCallО
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_3_14536207conv2d_3_14536209*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_145359052"
 conv2d_3/StatefulPartitionedCall
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_145359262
leaky_re_lu_5/PartitionedCall
dropout_4/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_145359512
dropout_4/PartitionedCallі
flatten_1/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_145359702
flatten_1/PartitionedCallВ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_14536215dense_2_14536217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_145359882!
dense_2/StatefulPartitionedCall
leaky_re_lu_6/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_145360092
leaky_re_lu_6/PartitionedCallњ
dropout_5/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_145360342
dropout_5/PartitionedCallВ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_3_14536222dense_3_14536224*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_145360572!
dense_3/StatefulPartitionedCall
leaky_re_lu_7/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_145360782
leaky_re_lu_7/PartitionedCallњ
reshape_1/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_145360982
reshape_1/PartitionedCall
IdentityIdentity"reshape_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
­
E__inference_dense_2_layer_call_and_return_conditional_losses_14536558

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
Ў
F__inference_conv2d_3_layer_call_and_return_conditional_losses_14535905

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
-

$__inference__traced_restore_14536743
file_prefix$
 assignvariableop_conv2d_2_kernel$
 assignvariableop_1_conv2d_2_bias&
"assignvariableop_2_conv2d_3_kernel$
 assignvariableop_3_conv2d_3_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias
assignvariableop_8_total
assignvariableop_9_count
identity_11ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ы
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*з
valueЭBЪB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesт
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4І
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Є
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Є
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpК
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ъ
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_14536527

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
БM
б
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536338

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpП
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_2/BiasAdd
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
leaky_re_lu_4/LeakyReluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_3/dropout/ConstИ
dropout_3/dropout/MulMul%leaky_re_lu_4/LeakyRelu:activations:0 dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeк
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2"
 dropout_3/dropout/GreaterEqual/yю
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2 
dropout_3/dropout/GreaterEqualЅ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/CastЊ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul_1А
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpд
conv2d_3/Conv2DConv2Ddropout_3/dropout/Mul_1:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_3/BiasAdd
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
alpha%>2
leaky_re_lu_5/LeakyReluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_4/dropout/ConstИ
dropout_4/dropout/MulMul%leaky_re_lu_5/LeakyRelu:activations:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_4/dropout/Mul
dropout_4/dropout/ShapeShape%leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shapeк
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2"
 dropout_4/dropout/GreaterEqual/yю
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2 
dropout_4/dropout/GreaterEqualЅ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout_4/dropout/CastЊ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_4/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_4/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_1/ReshapeЇ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dense_2/MatMulЅ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02 
dense_2/BiasAdd/ReadVariableOpЂ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dense_2/BiasAdd
leaky_re_lu_6/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџР*
alpha%>2
leaky_re_lu_6/LeakyReluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_5/dropout/ConstБ
dropout_5/dropout/MulMul%leaky_re_lu_6/LeakyRelu:activations:0 dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout_5/dropout/Mul
dropout_5/dropout/ShapeShape%leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shapeг
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2"
 dropout_5/dropout/GreaterEqual/yч
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџР2
dropout_5/dropout/CastЃ
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dropout_5/dropout/Mul_1Ї
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
Рс*
dtype02
dense_3/MatMul/ReadVariableOpЁ
dense_3/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
dense_3/MatMulЅ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:с*
dtype02 
dense_3/BiasAdd/ReadVariableOpЂ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
dense_3/BiasAdd
leaky_re_lu_7/LeakyRelu	LeakyReludense_3/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџс*
alpha%>2
leaky_re_lu_7/LeakyReluw
reshape_1/ShapeShape%leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicey
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :с2
reshape_1/Reshape/shape/1Ў
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape­
reshape_1/ReshapeReshape%leaky_re_lu_7/LeakyRelu:activations:0 reshape_1/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2
reshape_1/Reshapeo
IdentityIdentityreshape_1/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ:::::::::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
­
E__inference_dense_3_layer_call_and_return_conditional_losses_14536057

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Рс*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:с*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
г
ц
/__inference_sequential_1_layer_call_fn_14536249
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџс**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_145362302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџс2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv2d_2_input"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*У
serving_defaultЏ
Q
conv2d_2_input?
 serving_default_conv2d_2_input:0џџџџџџџџџ>
	reshape_11
StatefulPartitionedCall:0џџџџџџџџџсtensorflow/serving/predict:Еб
АG
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
#_self_saveable_object_factories
	optimizer

signatures
	variables
regularization_losses
trainable_variables
	keras_api
Љ_default_save_signature
Њ__call__
+Ћ&call_and_return_all_conditional_losses"ЉC
_tf_keras_sequentialC{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3136, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2401, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2401]}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3136, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2401, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2401]}}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.04, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}


kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses"Щ	
_tf_keras_layerЏ	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 1]}}

#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
 	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}

#!_self_saveable_object_factories
"	variables
#regularization_losses
$trainable_variables
%	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}


&kernel
'bias
#(_self_saveable_object_factories
)	variables
*regularization_losses
+trainable_variables
,	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"Ы	
_tf_keras_layerБ	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 16]}}

#-_self_saveable_object_factories
.	variables
/regularization_losses
0trainable_variables
1	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}

#2_self_saveable_object_factories
3	variables
4regularization_losses
5trainable_variables
6	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}

#7_self_saveable_object_factories
8	variables
9regularization_losses
:trainable_variables
;	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


<kernel
=bias
#>_self_saveable_object_factories
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3136, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 144]}}

#C_self_saveable_object_factories
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}

#H_self_saveable_object_factories
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
О__call__
+П&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}


Mkernel
Nbias
#O_self_saveable_object_factories
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2401, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3136}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3136]}}

#T_self_saveable_object_factories
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}

#Y_self_saveable_object_factories
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"х
_tf_keras_layerЫ{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2401]}}}
 "
trackable_dict_wrapper
"
	optimizer
-
Цserving_default"
signature_map
X
0
1
&2
'3
<4
=5
M6
N7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
&2
'3
<4
=5
M6
N7"
trackable_list_wrapper
Ю
^layer_regularization_losses
_metrics
	variables
`layer_metrics
anon_trainable_variables
regularization_losses

blayers
trainable_variables
Њ__call__
Љ_default_save_signature
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
clayer_regularization_losses
dmetrics
elayer_metrics
	variables
fnon_trainable_variables
regularization_losses

glayers
trainable_variables
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
hlayer_regularization_losses
imetrics
jlayer_metrics
	variables
knon_trainable_variables
regularization_losses

llayers
trainable_variables
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
mlayer_regularization_losses
nmetrics
olayer_metrics
"	variables
pnon_trainable_variables
#regularization_losses

qlayers
$trainable_variables
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_3/kernel
:2conv2d_3/bias
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
А
rlayer_regularization_losses
smetrics
tlayer_metrics
)	variables
unon_trainable_variables
*regularization_losses

vlayers
+trainable_variables
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
wlayer_regularization_losses
xmetrics
ylayer_metrics
.	variables
znon_trainable_variables
/regularization_losses

{layers
0trainable_variables
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Б
|layer_regularization_losses
}metrics
~layer_metrics
3	variables
non_trainable_variables
4regularization_losses
layers
5trainable_variables
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
metrics
layer_metrics
8	variables
non_trainable_variables
9regularization_losses
layers
:trainable_variables
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
": 
Р2dense_2/kernel
:Р2dense_2/bias
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
Е
 layer_regularization_losses
metrics
layer_metrics
?	variables
non_trainable_variables
@regularization_losses
layers
Atrainable_variables
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
metrics
layer_metrics
D	variables
non_trainable_variables
Eregularization_losses
layers
Ftrainable_variables
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
metrics
layer_metrics
I	variables
non_trainable_variables
Jregularization_losses
layers
Ktrainable_variables
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
": 
Рс2dense_3/kernel
:с2dense_3/bias
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
Е
 layer_regularization_losses
metrics
layer_metrics
P	variables
non_trainable_variables
Qregularization_losses
layers
Rtrainable_variables
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
metrics
layer_metrics
U	variables
non_trainable_variables
Vregularization_losses
layers
Wtrainable_variables
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
 metrics
Ёlayer_metrics
Z	variables
Ђnon_trainable_variables
[regularization_losses
Ѓlayers
\trainable_variables
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Є0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

Ѕtotal

Іcount
Ї	variables
Ј	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
Ѕ0
І1"
trackable_list_wrapper
.
Ї	variables"
_generic_user_object
№2э
#__inference__wrapped_model_14535822Х
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *5Ђ2
0-
conv2d_2_inputџџџџџџџџџ
2
/__inference_sequential_1_layer_call_fn_14536404
/__inference_sequential_1_layer_call_fn_14536249
/__inference_sequential_1_layer_call_fn_14536425
/__inference_sequential_1_layer_call_fn_14536195Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536383
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536338
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536107
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536140Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_conv2d_2_layer_call_fn_14536444Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_2_layer_call_and_return_conditional_losses_14536435Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_leaky_re_lu_4_layer_call_fn_14536454Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_14536449Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
,__inference_dropout_3_layer_call_fn_14536481
,__inference_dropout_3_layer_call_fn_14536476Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_dropout_3_layer_call_and_return_conditional_losses_14536466
G__inference_dropout_3_layer_call_and_return_conditional_losses_14536471Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_conv2d_3_layer_call_fn_14536500Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_3_layer_call_and_return_conditional_losses_14536491Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_leaky_re_lu_5_layer_call_fn_14536510Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_14536505Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
,__inference_dropout_4_layer_call_fn_14536537
,__inference_dropout_4_layer_call_fn_14536532Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_dropout_4_layer_call_and_return_conditional_losses_14536527
G__inference_dropout_4_layer_call_and_return_conditional_losses_14536522Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
,__inference_flatten_1_layer_call_fn_14536548Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_flatten_1_layer_call_and_return_conditional_losses_14536543Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_2_layer_call_fn_14536567Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_2_layer_call_and_return_conditional_losses_14536558Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_leaky_re_lu_6_layer_call_fn_14536577Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_14536572Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
,__inference_dropout_5_layer_call_fn_14536599
,__inference_dropout_5_layer_call_fn_14536604Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536589
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536594Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_dense_3_layer_call_fn_14536623Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_3_layer_call_and_return_conditional_losses_14536614Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_leaky_re_lu_7_layer_call_fn_14536633Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_14536628Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_reshape_1_layer_call_fn_14536650Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_reshape_1_layer_call_and_return_conditional_losses_14536645Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
<B:
&__inference_signature_wrapper_14536272conv2d_2_inputЋ
#__inference__wrapped_model_14535822&'<=MN?Ђ<
5Ђ2
0-
conv2d_2_inputџџџџџџџџџ
Њ "6Њ3
1
	reshape_1$!
	reshape_1џџџџџџџџџсЖ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_14536435l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_conv2d_2_layer_call_fn_14536444_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЖ
F__inference_conv2d_3_layer_call_and_return_conditional_losses_14536491l&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_conv2d_3_layer_call_fn_14536500_&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЇ
E__inference_dense_2_layer_call_and_return_conditional_losses_14536558^<=0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџР
 
*__inference_dense_2_layer_call_fn_14536567Q<=0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџРЇ
E__inference_dense_3_layer_call_and_return_conditional_losses_14536614^MN0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "&Ђ#

0џџџџџџџџџс
 
*__inference_dense_3_layer_call_fn_14536623QMN0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџсЗ
G__inference_dropout_3_layer_call_and_return_conditional_losses_14536466l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 З
G__inference_dropout_3_layer_call_and_return_conditional_losses_14536471l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 
,__inference_dropout_3_layer_call_fn_14536476_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџ
,__inference_dropout_3_layer_call_fn_14536481_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџЗ
G__inference_dropout_4_layer_call_and_return_conditional_losses_14536522l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 З
G__inference_dropout_4_layer_call_and_return_conditional_losses_14536527l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 
,__inference_dropout_4_layer_call_fn_14536532_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџ
,__inference_dropout_4_layer_call_fn_14536537_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџЉ
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536589^4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p
Њ "&Ђ#

0џџџџџџџџџР
 Љ
G__inference_dropout_5_layer_call_and_return_conditional_losses_14536594^4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p 
Њ "&Ђ#

0џџџџџџџџџР
 
,__inference_dropout_5_layer_call_fn_14536599Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p
Њ "џџџџџџџџџР
,__inference_dropout_5_layer_call_fn_14536604Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p 
Њ "џџџџџџџџџРЌ
G__inference_flatten_1_layer_call_and_return_conditional_losses_14536543a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_flatten_1_layer_call_fn_14536548T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџЗ
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_14536449h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
0__inference_leaky_re_lu_4_layer_call_fn_14536454[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЗ
K__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_14536505h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
0__inference_leaky_re_lu_5_layer_call_fn_14536510[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЉ
K__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_14536572Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "&Ђ#

0џџџџџџџџџР
 
0__inference_leaky_re_lu_6_layer_call_fn_14536577M0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџРЉ
K__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_14536628Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџс
Њ "&Ђ#

0џџџџџџџџџс
 
0__inference_leaky_re_lu_7_layer_call_fn_14536633M0Ђ-
&Ђ#
!
inputsџџџџџџџџџс
Њ "џџџџџџџџџсЅ
G__inference_reshape_1_layer_call_and_return_conditional_losses_14536645Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџс
Њ "&Ђ#

0џџџџџџџџџс
 }
,__inference_reshape_1_layer_call_fn_14536650M0Ђ-
&Ђ#
!
inputsџџџџџџџџџс
Њ "џџџџџџџџџсЩ
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536107{&'<=MNGЂD
=Ђ:
0-
conv2d_2_inputџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџс
 Щ
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536140{&'<=MNGЂD
=Ђ:
0-
conv2d_2_inputџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџс
 С
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536338s&'<=MN?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџс
 С
J__inference_sequential_1_layer_call_and_return_conditional_losses_14536383s&'<=MN?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџс
 Ё
/__inference_sequential_1_layer_call_fn_14536195n&'<=MNGЂD
=Ђ:
0-
conv2d_2_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџсЁ
/__inference_sequential_1_layer_call_fn_14536249n&'<=MNGЂD
=Ђ:
0-
conv2d_2_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџс
/__inference_sequential_1_layer_call_fn_14536404f&'<=MN?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџс
/__inference_sequential_1_layer_call_fn_14536425f&'<=MN?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџсР
&__inference_signature_wrapper_14536272&'<=MNQЂN
Ђ 
GЊD
B
conv2d_2_input0-
conv2d_2_inputџџџџџџџџџ"6Њ3
1
	reshape_1$!
	reshape_1џџџџџџџџџс