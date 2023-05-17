/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated by lcm-gen
 **/

#ifndef __lcmcomm_HighCmd_hpp__
#define __lcmcomm_HighCmd_hpp__

#include <lcm/lcm_coretypes.h>

#include "lcmcomm/LED.hpp"

namespace lcmcomm
{

/// high level feedback
class HighCmd
{
    public:
        uint8_t    levelFlag;

        int16_t    commVersion;

        int16_t    robotID;

        int32_t    SN;

        uint8_t    bandWidth;

        uint8_t    mode;

        /// 0:idle, default stand      1:forced stand     2:walk continuously
        float      forwardSpeed;

        /// speed of move forward or backward, scale: -1~1
        float      sideSpeed;

        /// speed of move left or right, scale: -1~1
        float      rotateSpeed;

        /// speed of spin left or right, scale: -1~1
        float      bodyHeight;

        /// body height, scale: -1~1
        float      footRaiseHeight;

        /// foot up height while walking (unavailable now)
        float      yaw;

        /// unit: radian, scale: -1~1
        float      pitch;

        /// unit: radian, scale: -1~1
        float      roll;

        /// unit: radian, scale: -1~1
        lcmcomm::LED led[4];

        uint8_t    wirelessRemote[40];

        uint8_t    AppRemote[40];

        int32_t    reserve;

        int32_t    crc;

    public:
        /**
         * Encode a message into binary form.
         *
         * @param buf The output buffer.
         * @param offset Encoding starts at thie byte offset into @p buf.
         * @param maxlen Maximum number of bytes to write.  This should generally be
         *  equal to getEncodedSize().
         * @return The number of bytes encoded, or <0 on error.
         */
        inline int encode(void *buf, int offset, int maxlen) const;

        /**
         * Check how many bytes are required to encode this message.
         */
        inline int getEncodedSize() const;

        /**
         * Decode a message from binary form into this instance.
         *
         * @param buf The buffer containing the encoded message.
         * @param offset The byte offset into @p buf where the encoded message starts.
         * @param maxlen The maximum number of bytes to read while decoding.
         * @return The number of bytes decoded, or <0 if an error occured.
         */
        inline int decode(const void *buf, int offset, int maxlen);

        /**
         * Retrieve the 64-bit fingerprint identifying the structure of the message.
         * Note that the fingerprint is the same for all instances of the same
         * message type, and is a fingerprint on the message type definition, not on
         * the message contents.
         */
        inline static int64_t getHash();

        /**
         * Returns "HighCmd"
         */
        inline static const char* getTypeName();

        // LCM support functions. Users should not call these
        inline int _encodeNoHash(void *buf, int offset, int maxlen) const;
        inline int _getEncodedSizeNoHash() const;
        inline int _decodeNoHash(const void *buf, int offset, int maxlen);
        inline static uint64_t _computeHash(const __lcm_hash_ptr *p);
};

int HighCmd::encode(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;
    int64_t hash = getHash();

    tlen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
    if (tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int HighCmd::decode(const void *buf, int offset, int maxlen)
{
    int pos = 0, thislen;

    int64_t msg_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (msg_hash != getHash()) return -1;

    thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int HighCmd::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t HighCmd::getHash()
{
    static int64_t hash = static_cast<int64_t>(_computeHash(NULL));
    return hash;
}

const char* HighCmd::getTypeName()
{
    return "HighCmd";
}

int HighCmd::_encodeNoHash(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;

    tlen = __byte_encode_array(buf, offset + pos, maxlen - pos, &this->levelFlag, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int16_t_encode_array(buf, offset + pos, maxlen - pos, &this->commVersion, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int16_t_encode_array(buf, offset + pos, maxlen - pos, &this->robotID, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->SN, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __byte_encode_array(buf, offset + pos, maxlen - pos, &this->bandWidth, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __byte_encode_array(buf, offset + pos, maxlen - pos, &this->mode, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->forwardSpeed, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->sideSpeed, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->rotateSpeed, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->bodyHeight, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->footRaiseHeight, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->yaw, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->pitch, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->roll, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    for (int a0 = 0; a0 < 4; a0++) {
        tlen = this->led[a0]._encodeNoHash(buf, offset + pos, maxlen - pos);
        if(tlen < 0) return tlen; else pos += tlen;
    }

    tlen = __byte_encode_array(buf, offset + pos, maxlen - pos, &this->wirelessRemote[0], 40);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __byte_encode_array(buf, offset + pos, maxlen - pos, &this->AppRemote[0], 40);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->reserve, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->crc, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int HighCmd::_decodeNoHash(const void *buf, int offset, int maxlen)
{
    int pos = 0, tlen;

    tlen = __byte_decode_array(buf, offset + pos, maxlen - pos, &this->levelFlag, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int16_t_decode_array(buf, offset + pos, maxlen - pos, &this->commVersion, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int16_t_decode_array(buf, offset + pos, maxlen - pos, &this->robotID, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->SN, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __byte_decode_array(buf, offset + pos, maxlen - pos, &this->bandWidth, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __byte_decode_array(buf, offset + pos, maxlen - pos, &this->mode, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->forwardSpeed, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->sideSpeed, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->rotateSpeed, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->bodyHeight, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->footRaiseHeight, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->yaw, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->pitch, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->roll, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    for (int a0 = 0; a0 < 4; a0++) {
        tlen = this->led[a0]._decodeNoHash(buf, offset + pos, maxlen - pos);
        if(tlen < 0) return tlen; else pos += tlen;
    }

    tlen = __byte_decode_array(buf, offset + pos, maxlen - pos, &this->wirelessRemote[0], 40);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __byte_decode_array(buf, offset + pos, maxlen - pos, &this->AppRemote[0], 40);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->reserve, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->crc, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int HighCmd::_getEncodedSizeNoHash() const
{
    int enc_size = 0;
    enc_size += __byte_encoded_array_size(NULL, 1);
    enc_size += __int16_t_encoded_array_size(NULL, 1);
    enc_size += __int16_t_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += __byte_encoded_array_size(NULL, 1);
    enc_size += __byte_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    enc_size += __float_encoded_array_size(NULL, 1);
    for (int a0 = 0; a0 < 4; a0++) {
        enc_size += this->led[a0]._getEncodedSizeNoHash();
    }
    enc_size += __byte_encoded_array_size(NULL, 40);
    enc_size += __byte_encoded_array_size(NULL, 40);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    return enc_size;
}

uint64_t HighCmd::_computeHash(const __lcm_hash_ptr *p)
{
    const __lcm_hash_ptr *fp;
    for(fp = p; fp != NULL; fp = fp->parent)
        if(fp->v == HighCmd::getHash)
            return 0;
    const __lcm_hash_ptr cp = { p, HighCmd::getHash };

    uint64_t hash = 0x059bc62cf3bd3093LL +
         lcmcomm::LED::_computeHash(&cp);

    return (hash<<1) + ((hash>>63)&1);
}

}

#endif