/*=============================================================================

	ReShade 4 effect file
    github.com/martymcmodding

	Support me:
   		paypal.me/mcflypg
   		patreon.com/mcflypg

    Ray Traced Screen Space Global Illumination 

    by Marty McFly / P.Gilcher
    part of qUINT shader library for ReShade 4

    CC BY-NC-ND 3.0 licensed.

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

//these two are required if I want to reuse the MXAO textures properly

#ifndef MXAO_MIPLEVEL_AO
 #define MXAO_MIPLEVEL_AO		0	//[0 to 2]      Miplevel of AO texture. 0 = fullscreen, 1 = 1/2 screen width/height, 2 = 1/4 screen width/height and so forth. Best results: IL MipLevel = AO MipLevel + 2
#endif

#ifndef MXAO_MIPLEVEL_IL
 #define MXAO_MIPLEVEL_IL		2	//[0 to 4]      Miplevel of IL texture. 0 = fullscreen, 1 = 1/2 screen width/height, 2 = 1/4 screen width/height and so forth.
#endif

#ifndef INFINITE_BOUNCES
 #define INFINITE_BOUNCES       0   //[0 or 1]      If enabled, path tracer samples previous frame GI as well, causing a feedback loop to simulate secondary bounces, causing a more widespread GI.
#endif

#ifndef SPATIAL_FILTER
 #define SPATIAL_FILTER	       	1   //[0 or 1]      If enabled, final GI is filtered for a less noisy but also less precise result. Enabled by default.
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float RT_SAMPLE_RADIUS <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 20.0;
    ui_step = 0.01;
    ui_label = "Ray Length";
	ui_tooltip = "Maximum ray length, directly affects\nthe spread radius of shadows / indirect lighing";
    ui_category = "Path Tracing";
> = 15.0;

uniform int RT_RAY_AMOUNT <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Ray Amount";
    ui_category = "Path Tracing";
> = 10;

uniform int RT_RAY_STEPS <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Ray Step Amount";
    ui_category = "Path Tracing";
> = 10;

uniform float RT_AO_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 2.0;
    ui_step = 0.01;
    ui_label = "Ambient Occlusion Intensity";
    ui_category = "Blending";
> = 1.0;

uniform float RT_IL_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Indirect Lighting Intensity";
    ui_category = "Blending";
> = 4.0;

#if INFINITE_BOUNCES != 0
    uniform float RT_IL_BOUNCE_WEIGHT <
        ui_type = "drag";
        ui_min = 0; ui_max = 5.0;
        ui_step = 0.01;
        ui_label = "Next Bounce Weight";
        ui_category = "Blending";
    > = 0.0;
#endif

uniform float2 RT_FADE_DEPTH <
	ui_type = "drag";
    ui_label = "Fade Out Start / End";
	ui_min = 0.00; ui_max = 1.00;
	ui_tooltip = "Distance where GI starts to fade out | is completely faded out.";
    ui_category = "Blending";
> = float2(0.0, 0.5);

uniform int RT_DEBUG_VIEW <
	ui_type = "combo";
    ui_label = "Enable Debug View";
	ui_items = "None\0AO/IL channel\0";
	ui_tooltip = "Different debug outputs";
    ui_category = "Debug";
> = 0;
/*
uniform float3 SKY_COLOR <
	ui_type = "color";
	ui_label = "Sky Color";
> = float3(1.0, 0.0, 0.0);


uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);*/


/*=============================================================================
	Textures, Samplers, Globals
=============================================================================*/

#include "qUINT_common.fxh"

uniform int framecount < source = "framecount"; >;

texture2D MXAO_ColorTex 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA8; MipLevels = 3+MXAO_MIPLEVEL_IL;};
texture2D MXAO_DepthTex 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;  MipLevels = 3+MXAO_MIPLEVEL_AO;};
texture2D MXAO_NormalTex	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA8; MipLevels = 3+MXAO_MIPLEVEL_IL;};

texture2D GITex	            { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = 4;};
texture2D GBufferTex	    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture2D GITexPrev	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture2D GBufferTexPrev    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };

texture JitterTex           < source = "LDR_RGB1_18.png"; > { Width = 32; Height = 32; Format = RGBA8; };

sampler2D sMXAO_ColorTex	{ Texture = MXAO_ColorTex;	};
sampler2D sMXAO_DepthTex	{ Texture = MXAO_DepthTex;	};
sampler2D sMXAO_NormalTex	{ Texture = MXAO_NormalTex;	};

sampler2D sGITex	        { Texture = GITex;	};
sampler2D sGBufferTex	    { Texture = GBufferTex;	};
sampler2D sGITexPrev	    { Texture = GITexPrev;	};
sampler2D sGBufferTexPrev	{ Texture = GBufferTexPrev;	};

sampler	sJitterTex        { Texture = JitterTex; AddressU = WRAP; AddressV = WRAP;};

/*=============================================================================
	Vertex Shader
=============================================================================*/

struct VSOUT
{
	float4                  vpos        : SV_Position;
    float2                  uv          : TEXCOORD0;
    nointerpolation float3  uvtoviewADD : TEXCOORD1;
    nointerpolation float3  uvtoviewMUL : TEXCOORD2;
    nointerpolation float4  viewtouv    : TEXCOORD3;
};

VSOUT VS_RT(in uint id : SV_VertexID)
{
    VSOUT o;

    o.uv.x = (id == 2) ? 2.0 : 0.0;
    o.uv.y = (id == 1) ? 2.0 : 0.0;

    o.vpos = float4(o.uv.xy * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    
    o.uvtoviewADD = float3(-1.0,-1.0,1.0);
    o.uvtoviewMUL = float3(2.0,2.0,0.0);

#if 1
    static const float FOV = 75; //vertical FoV
    o.uvtoviewADD = float3(-tan(radians(FOV * 0.5)).xx,1.0) * qUINT::ASPECT_RATIO.yxx;
   	o.uvtoviewMUL = float3(-2.0 * o.uvtoviewADD.xy,0.0);
#endif

	o.viewtouv.xy = rcp(o.uvtoviewMUL.xy);
    o.viewtouv.zw = -o.uvtoviewADD.xy * o.viewtouv.xy;

    return o;
}

/*=============================================================================
	Functions
=============================================================================*/

struct Ray 
{
    float3 pos;
    float3 dir;
    float2 uv;
};

struct MRT
{
    float4 gi   : SV_Target0;
    float4 gbuf : SV_Target1;
};

float3 get_position_from_uv(in VSOUT i)
{
    return (i.uv.xyx * i.uvtoviewMUL + i.uvtoviewADD) * qUINT::linear_depth(i.uv.xy) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
}

float3 get_position_from_uv(in VSOUT i, in float2 uv)
{
    return (uv.xyx * i.uvtoviewMUL + i.uvtoviewADD) * qUINT::linear_depth(uv) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
}

float3 get_position_from_uv(in VSOUT i, in float2 uv, in int mip)
{
    return (uv.xyx * i.uvtoviewMUL + i.uvtoviewADD) * tex2Dlod(sMXAO_DepthTex, float4(uv.xyx, mip)).x;
}

float2 get_uv_from_position(in VSOUT i, in float3 pos)
{
	return (pos.xy / pos.z) * i.viewtouv.xy + i.viewtouv.zw;
}

float3x3 get_tbn(float3 n)
{
    float3 temp = float3(0.707,0.707,0);
	temp = lerp(temp, temp.zxy, saturate(1 - 10 * dot(temp, n)));
	float3 t = normalize(cross(temp, n));
	float3 b = cross(n,t);
	return float3x3(t,b,n);
}

void unpack_hdr(inout float3 color)
{
    color /= 1.01 - color; //min(min(color.r, color.g), color.b);//max(max(color.r, color.g), color.b);
}

void pack_hdr(inout float3 color)
{
    color /= 1.01 + color; //min(min(color.r, color.g), color.b);//max(max(color.r, color.g), color.b);
}

float get_blend_weight(MRT curr, MRT prev)
{
	float4 gbuff_delta = abs(curr.gbuf - prev.gbuf);

    float normal_diff = max(max(gbuff_delta.x, 
    							gbuff_delta.y), 
    							gbuff_delta.z); 

    float depth_diff = gbuff_delta.w;

    float alpha = normal_diff * 10 * saturate(depth_diff * 10000 + 1) + depth_diff * 5000;

    alpha = alpha / (1 + alpha);
    alpha = lerp(0.03, 0.9, alpha);
    
    return alpha;
}

/*=============================================================================
	Pixel Shaders
=============================================================================*/

void PS_InputBufferSetup(in VSOUT i, out float4 color : SV_Target0, out float4 depth : SV_Target1, out float4 normal : SV_Target2)
{
    float3 delta = float3(qUINT::PIXEL_SIZE.xy, 0);

	float3 pos          =         get_position_from_uv(i, i.uv.xy);
	float3 pos_delta_x1 = - pos + get_position_from_uv(i, i.uv.xy + delta.xz);
	float3 pos_delta_x2 =   pos - get_position_from_uv(i, i.uv.xy - delta.xz);
	float3 pos_delta_y1 = - pos + get_position_from_uv(i, i.uv.xy + delta.zy);
	float3 pos_delta_y2 =   pos - get_position_from_uv(i, i.uv.xy - delta.zy);

	pos_delta_x1 = lerp(pos_delta_x1, pos_delta_x2, abs(pos_delta_x1.z) > abs(pos_delta_x2.z));
	pos_delta_y1 = lerp(pos_delta_y1, pos_delta_y2, abs(pos_delta_y1.z) > abs(pos_delta_y2.z));

	normal  = float4(normalize(cross(pos_delta_y1, pos_delta_x1)) * 0.5 + 0.5, 1);
    color 	= tex2D(qUINT::sBackBufferTex, i.uv);
	depth 	= qUINT::linear_depth(i.uv) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 0.1;   
}

void PS_StencilSetup(in VSOUT i, out float4 o : SV_Target0)
{   
    o = 0;

    if(qUINT::linear_depth(i.uv.xy) >= max(RT_FADE_DEPTH.x, RT_FADE_DEPTH.y) //theoretically only .y but users might swap it...
    ) discard;    
}

void PS_RTMain(in VSOUT i, out MRT o)
{
    int NUM_STEPS  = RT_RAY_STEPS;
    int NUM_RAYS =  RT_RAY_AMOUNT;

    float3      position = get_position_from_uv(i);
    float3      normal   = tex2D(sMXAO_NormalTex, i.uv.xy).xyz * 2 - 1;
    float3x3    tbn      = get_tbn(normal);
    float3      jitter   = frac(tex2Dfetch(sJitterTex, int4(i.vpos.xy % tex2Dsize(sJitterTex), 0, 0)).xyz + (framecount  % 1000) * 3.1 /* no, not pi */);    

    float depth = position.z / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    position = position * 0.998 + normal.xyz * depth;

    float2 sample_dir;
    sincos(38.39941 * jitter.x, sample_dir.x, sample_dir.y); //2.3999632 * 16 

    MRT curr, prev;
    curr.gbuf = float4(normal, depth);  
    prev.gi = tex2D(sGITexPrev, i.uv.xy);
    prev.gbuf = tex2D(sGBufferTexPrev, i.uv.xy); 
    float alpha = get_blend_weight(curr, prev);

    NUM_RAYS += 15 * alpha; //drastically increase quality on areas where temporal filter fails   

    float ray_step = RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS;
    float thickness = 1.0 / (RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS);

    curr.gi = 0;

    [loop]
    for(float r = 0; r < NUM_RAYS; r++)
    {
        Ray ray;
        ray.dir.z = (r + jitter.y) / NUM_RAYS;
        ray.dir.xy = sample_dir * sqrt(1 - ray.dir.z * ray.dir.z);
        ray.dir = mul(ray.dir, tbn);
        sample_dir = mul(sample_dir, float2x2(0.76465, -0.64444, 0.64444, 0.76465)); 

        float intersected = 0, mip = 0; int s = 0; bool inside_screen = 1;

        while(s++ < NUM_STEPS && inside_screen)
        {
            float lambda = float(s - jitter.z) / NUM_STEPS; //normalized position in ray [0, 1]
            lambda *= lambda * rsqrt(lambda); //lambda ^ 1.5 using the fastest instruction sets

            ray.pos = position + ray.dir * lambda * ray_step;

            ray.uv = get_uv_from_position(i, ray.pos);
            inside_screen = all(saturate(-ray.uv * ray.uv + ray.uv));

            mip = length((ray.uv - i.uv.xy) * qUINT::ASPECT_RATIO.yx) * 28;
            float3 delta = get_position_from_uv(i, ray.uv, mip + MXAO_MIPLEVEL_AO) - ray.pos;
            
            delta *= thickness;

            [branch]
            if(delta.z < 0 && delta.z > -1)
            {                
                intersected = saturate(1 - dot(delta, delta)) * inside_screen; 
                s = NUM_STEPS;
            }
        }         

        curr.gi.w += intersected;        

        [branch]
        if(RT_IL_AMOUNT > 0 && intersected > 0.05)
        {
            float3 albedo 			= tex2Dlod(sMXAO_ColorTex, 	float4(ray.uv, 0, mip + MXAO_MIPLEVEL_IL)).rgb; unpack_hdr(albedo);
            float3 intersect_normal = tex2Dlod(sMXAO_NormalTex, float4(ray.uv, 0, mip + MXAO_MIPLEVEL_IL)).xyz * 2 - 1;

 #if INFINITE_BOUNCES != 0
            float3 nextbounce 		= tex2Dlod(sGITexPrev, float4(ray.uv, 0, 0)).rgb; unpack_hdr(nextbounce);            
            albedo += nextbounce * RT_IL_BOUNCE_WEIGHT;
#endif
            curr.gi.rgb += albedo * intersected * saturate(dot(-intersect_normal, ray.dir));
        }
    }

    //even though the math does not allow for it, NUM_RAYS can be 0
    //occasionally and cause NaN's all over the place.
    curr.gi /=  NUM_RAYS + 1e-6; 
    pack_hdr(curr.gi.rgb);

    o.gi = lerp(prev.gi, curr.gi, alpha);
    o.gbuf = curr.gbuf;
}

void PS_CopyAndFilter(in VSOUT i, out MRT o)
{
	o.gi    = tex2D(sGITex,      i.uv.xy);
    o.gbuf  = tex2D(sGBufferTex, i.uv.xy);
#if SPATIAL_FILTER != 0
    float jitter = dot(floor(i.vpos.xy % 4 + 0.1), float2(0.0625, 0.25)) + 0.0625;
    float2 dir; sincos(2.3999632 * 16 * jitter, dir.x, dir.y);

    float4 gi = o.gi;
    float weightsum = 1;

    [loop]
    for(int j = 0; j < 16; j++)
    {
        float2 sample_uv = i.uv.xy + dir * qUINT::PIXEL_SIZE * sqrt(j + jitter);   
        dir.xy = mul(dir.xy, float2x2(0.76465, -0.64444, 0.64444, 0.76465)); 

        float4 gi_tap   = tex2Dlod(sGITex,      float4(sample_uv, 0, 0));
        float4 gbuf_tap = tex2Dlod(sGBufferTex, float4(sample_uv, 0, 0));
        
        float4 gi_delta     = abs(gi_tap - o.gi);
        float4 gbuf_delta   = abs(gbuf_tap - o.gbuf);

        float ddepth    = gbuf_delta.w;
        float dnormal   = max(max(gbuf_delta.x, gbuf_delta.y), gbuf_delta.z);
        float dvalue    = dot(gi_delta, gi_delta);

        float w = exp(-ddepth * 750)
                * exp(-dnormal * 50 * saturate(ddepth * 10000))
                * exp(-dvalue * 150);

        gi += gi_tap * w;
        weightsum += w;
    }

    gi /= weightsum;
    o.gi = gi;
#endif 
}

//need this as backbuffer is not guaranteed to have RGBA8
void PS_Output(in VSOUT i, out float4 o : SV_Target0)
{
    float4 gi = tex2D(sGITexPrev, i.uv.xy);
    float3 color = tex2D(sMXAO_ColorTex, i.uv.xy).rgb;

    unpack_hdr(color);
    unpack_hdr(gi.rgb);

    gi *= smoothstep(RT_FADE_DEPTH.y, RT_FADE_DEPTH.x, qUINT::linear_depth(i.uv.xy));

    gi.w = RT_AO_AMOUNT > 1 ? pow(1 - gi.w, RT_AO_AMOUNT) : 1 - gi.w * RT_AO_AMOUNT;
    gi.rgb *= RT_IL_AMOUNT * RT_IL_AMOUNT;

    color = color * gi.w * (1 + gi.rgb);  

    if(RT_DEBUG_VIEW == 1)
        color.rgb = gi.w * (1 + gi.rgb);

    pack_hdr(color.rgb);
    o = float4(color, 1);
}

/*=============================================================================
	Techniques
=============================================================================*/



technique RT
{
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_InputBufferSetup;
		RenderTarget0 = MXAO_ColorTex;
		RenderTarget1 = MXAO_DepthTex;
		RenderTarget2 = MXAO_NormalTex;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_StencilSetup;
        ClearRenderTargets = true;
		StencilEnable = true;
		StencilPass = REPLACE;
        StencilRef = 1;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_RTMain;
        RenderTarget0 = GITex;
        RenderTarget1 = GBufferTex;
        ClearRenderTargets = true;
        StencilEnable = true;
        StencilPass = KEEP;
        StencilFunc = EQUAL;
        StencilRef = 1;
	}
        pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_CopyAndFilter;
        RenderTarget0 = GITexPrev;
        RenderTarget1 = GBufferTexPrev;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Output;
	}
}